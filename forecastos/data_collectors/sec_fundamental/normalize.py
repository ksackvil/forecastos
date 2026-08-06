"""Putting extracted statements onto a comparable timeline, and merging them.

Filings do not report comparable periods. An income statement in a 10-K covers
the full year, so the fourth quarter only exists as the year minus the three
quarters already filed. Cash flow statements are frequently cumulative from the
start of the fiscal year, so Q3 as filed is really nine months. Balance sheets
are instants, so a period's opening position is the previous filing's close.

All three repairs need the same thing: for a given row, the same company's
filing from N months earlier. That lookup used to run as a scan of the whole
frame per row - which made the pipeline quadratic in row count, and made adding
companies far more expensive than it should be. `adjacent` does it as one
sorted merge per offset instead, and the comparisons never leave a CIK.
"""

import numpy as np
import pandas as pd

# How far a filing may sit from where the offset puts it and still be taken as
# the one we were looking for. Fiscal calendars drift by a few days a year and
# filing dates move around holidays, so an exact date would match almost
# nothing.
ADJACENCY_TOLERANCE = pd.Timedelta(days=30)

# A period is taken as quarterly or annual if it lands within this many months
# of 3 or 12. Reported periods are rarely exactly that long.
PERIOD_TOLERANCE = 1.0

# Columns identifying a row, kept alongside the datapoints.
FLOW_KEYS = ['cik', 'accn', 'fy', 'fp', 'form',
             'start', 'end', 'period', 'filed', 'frame']
PIT_KEYS = ['cik', 'accn', 'fy', 'fp', 'form', 'end', 'filed', 'frame']

# What makes a reported figure distinct, used to drop rows that a company
# stubbed correctly and we then stubbed again.
DEDUPE_KEYS = ['cik', 'fy', 'fp', 'form',
               'accn', 'start', 'end', 'period', 'filed']

# Income and cash flow describe the same period, so they join on all of it.
FLOW_MERGE_KEYS = ['cik', 'fy', 'fp', 'form',
                   'accn', 'start', 'end', 'period', 'filed']

# The balance sheet is an instant, so it joins on the period's close only.
PIT_MERGE_KEYS = ['cik', 'fy', 'form', 'accn', 'end', 'filed']


def normalize(wide: pd.DataFrame, statement) -> pd.DataFrame:
    """Reduce an extracted statement to keys and datapoints, then repair periods."""
    keys = PIT_KEYS if statement.is_pit else FLOW_KEYS
    names = [dp.name for dp in statement.mappings]
    wide = wide[[c for c in keys + names if c in wide.columns]].copy()

    if wide.empty:
        return wide

    for step in _steps(statement):
        wide = step(wide, statement)

    return wide


def _steps(statement) -> list:
    """The repairs this statement needs, in the order they have to happen.

    Cash flow stubs its cumulative periods before implying Q4, because the Q4
    arithmetic subtracts three quarters and they have to be quarters first.
    """
    if statement.is_pit:
        return [_to_start_end_columns, _imply_starting_values]

    per_statement = {
        'income_statement': [_add_period, _keep_quarterly_and_annual, _imply_q4],
        'cashflow_statement': [
            _add_period, _imply_quarterly, _imply_q4, _keep_quarterly_and_annual],
    }
    return per_statement.get(
        statement.name, [_add_period, _keep_quarterly_and_annual])


# === Finding the filing N months back === #


def adjacent(df: pd.DataFrame, months: int, value_cols: list) -> pd.DataFrame:
    """For each row, the same company's filing about `months` earlier.

    Args:
        df: rows to look back from. Needs cik, end, filed.
        months: how far back to look.
        value_cols: columns to carry over from the matched row.

    Returns:
        A frame on `df`'s index holding the matched row's end, filed and
        `value_cols`, all null where nothing matched.
    """
    out_cols = ['end', 'filed'] + value_cols
    if df.empty:
        return pd.DataFrame(index=df.index, columns=out_cols, dtype='float64')

    position = _match_positions(df, months)
    found = position >= 0

    matched = {}
    for col in ('end', 'filed'):
        values = np.full(len(df), np.datetime64('NaT'), dtype='datetime64[ns]')
        values[found] = df[col].to_numpy()[position[found]]
        matched[col] = values
    for col in value_cols:
        values = np.full(len(df), np.nan)
        values[found] = df[col].to_numpy(dtype='float64')[position[found]]
        matched[col] = values

    return pd.DataFrame(matched, index=df.index)


def _match_positions(df: pd.DataFrame, months: int) -> np.ndarray:
    """Row offset of each row's counterpart `months` back, or -1 if there is none.

    Both the period end and the filing date have to line up. That pair is what
    distinguishes the prior quarter's own filing from the same quarter carried
    as a comparative in a later one - the period end matches either way, and
    only the filing date tells them apart.

    Neither date is unique on its own, so the candidates are found by bucketing
    period ends into tolerance-wide bins and joining a row against the three
    bins its target could fall in. That keeps the comparisons to a handful per
    row within one company, where scanning the frame per row - what this
    replaces - made the whole pipeline quadratic in row count.
    """
    n = len(df)
    end, filed = df['end'], df['filed']
    target_end = end - pd.DateOffset(months=months)
    target_filed = filed - pd.DateOffset(months=months)

    usable = end.notna() & filed.notna()
    left = pd.DataFrame({
        '_row': np.arange(n)[usable.to_numpy()],
        'cik': df.loc[usable, 'cik'].to_numpy(),
        'target_end': target_end[usable].to_numpy(),
        'target_filed': target_filed[usable].to_numpy(),
    })
    right = pd.DataFrame({
        '_pos': np.arange(n)[usable.to_numpy()],
        'cik': df.loc[usable, 'cik'].to_numpy(),
        'match_end': end[usable].to_numpy(),
        'match_filed': filed[usable].to_numpy(),
    })

    out = np.full(n, -1, dtype='int64')
    if left.empty:
        return out

    width = ADJACENCY_TOLERANCE.value
    right['_bin'] = right['match_end'].astype('int64') // width
    target_bin = left['target_end'].astype('int64') // width

    # A target within one tolerance of a bin edge lands in the neighbouring
    # bin, so all three have to be probed to cover the window.
    candidates = pd.concat(
        [left.assign(_bin=target_bin + offset).merge(
            right, on=['cik', '_bin'], how='inner')
         for offset in (-1, 0, 1)],
        ignore_index=True,
    )
    if candidates.empty:
        return out

    within = (
        (candidates['match_end'] - candidates['target_end']).abs()
        .lt(ADJACENCY_TOLERANCE)
        & (candidates['match_filed'] - candidates['target_filed']).abs()
        .lt(ADJACENCY_TOLERANCE)
    )
    candidates = candidates[within]

    # An ambiguous window is not resolved by picking one: two filings both
    # sitting where the prior quarter should be means we cannot say which is
    # the quarter, and a wrong pick would silently corrupt the arithmetic that
    # subtracts it.
    unique = candidates.groupby('_row')['_pos'].transform('size').eq(1)
    candidates = candidates[unique]

    out[candidates['_row'].to_numpy()] = candidates['_pos'].to_numpy()
    return out


# === Period repairs === #


def _add_period(df: pd.DataFrame, statement=None) -> pd.DataFrame:
    """Length of each row's period, in months."""
    df['period'] = _months_between(df['start'], df['end'])
    return df


def _keep_quarterly_and_annual(df: pd.DataFrame, statement=None) -> pd.DataFrame:
    """Drop periods that are neither a quarter nor a year.

    Filings carry plenty of other spans - six month stubs, transition periods,
    life-to-date totals - and none of them line up with anything else.
    """
    period = df['period']
    return df[(period - 3.0).abs().lt(PERIOD_TOLERANCE)
              | (period - 12.0).abs().lt(PERIOD_TOLERANCE)]


def _imply_quarterly(df: pd.DataFrame, statement) -> pd.DataFrame:
    """Turn cumulative year-to-date figures into the quarter alone.

    Q2 and Q3 cash flow statements usually run from the start of the fiscal
    year, so the quarter is the filing minus the one before it.
    """
    names = _value_columns(df, statement)
    prior = adjacent(df, 3, names)

    cumulative = (
        df['fp'].ne('FY')
        & df['fp'].ne('Q1')
        & (df['period'] - 3.0).abs().gt(PERIOD_TOLERANCE)
        & prior['end'].notna()
    )

    if cumulative.any():
        for name in names:
            df.loc[cumulative, name] = (
                df.loc[cumulative, name] - prior.loc[cumulative, name])
        df.loc[cumulative, 'start'] = (
            prior.loc[cumulative, 'end'] + pd.Timedelta(days=1))
        df.loc[cumulative, 'period'] = _months_between(
            df.loc[cumulative, 'start'], df.loc[cumulative, 'end'])

    # Some companies report the stubbed figure themselves, in which case our
    # stub reproduces a row that is already there.
    return df.drop_duplicates(subset=DEDUPE_KEYS, keep='first')


def _imply_q4(df: pd.DataFrame, statement) -> pd.DataFrame:
    """Add a fourth quarter, which no filing reports on its own.

    The 10-K covers the full year, so Q4 is that year less the three quarters
    already filed - and only where all three were found, since a missing one
    would silently turn into a year-sized quarter.
    """
    names = _value_columns(df, statement)
    q3, q2, q1 = (adjacent(df, months, names) for months in (3, 6, 9))

    full_year = (
        df['fp'].eq('FY')
        & (df['period'] - 12.0).abs().lt(PERIOD_TOLERANCE)
        & q1['end'].notna() & q2['end'].notna() & q3['end'].notna()
    )
    if not full_year.any():
        return df

    fourth = df.loc[full_year].copy()
    fourth['fp'] = 'Q4'
    fourth['start'] = q3.loc[full_year, 'end'] + pd.Timedelta(days=1)
    fourth['period'] = _months_between(fourth['start'], fourth['end'])
    # SEC's calendar label belongs to the period the filing reported, not to
    # one we derived from it.
    fourth['frame'] = np.nan

    for name in names:
        fourth[name] = df.loc[full_year, name] - (
            q3.loc[full_year, name]
            + q2.loc[full_year, name]
            + q1.loc[full_year, name])

    return pd.concat([df, fourth], ignore_index=True)


def _to_start_end_columns(df: pd.DataFrame, statement) -> pd.DataFrame:
    """Rename instant values to the period's close and make room for its open."""
    names = [dp.name for dp in statement.mappings if dp.name in df.columns]
    df = df.rename(columns={name: f'end_{name}' for name in names})
    for name in names:
        df[f'start_{name}'] = np.nan
    return df


def _imply_starting_values(df: pd.DataFrame, statement) -> pd.DataFrame:
    """Carry the previous filing's closing position in as this period's opening.

    An annual row opens where the prior year closed; a quarterly row opens
    where the prior quarter did.
    """
    names = [dp.name for dp in statement.mappings
             if f'end_{dp.name}' in df.columns]
    end_cols = [f'end_{name}' for name in names]
    if not end_cols:
        return df

    annual = adjacent(df, 12, end_cols)
    quarterly = adjacent(df, 3, end_cols)
    is_annual = df['fp'].eq('FY').to_numpy()

    for name in names:
        df[f'start_{name}'] = np.where(
            is_annual,
            annual[f'end_{name}'].to_numpy(dtype='float64'),
            quarterly[f'end_{name}'].to_numpy(dtype='float64'))

    return df


# === Merging === #


def merge_statements(statements: dict) -> pd.DataFrame:
    """One row per company-period, carrying every statement's datapoints.

    Args:
        statements: normalized frames keyed by statement name.

    Returns:
        The merged frame, or an empty one if there was nothing to join.
    """
    income = statements.get('income_statement')
    cashflow = statements.get('cashflow_statement')
    balance = statements.get('balance_sheet')

    if income is None or income.empty or cashflow is None or cashflow.empty:
        return pd.DataFrame()

    merged = income.merge(
        # `frame` is the period's calendar label and both sides carry the same
        # one, so it is taken from the income statement alone.
        cashflow.drop(columns=['frame'], errors='ignore'),
        on=FLOW_MERGE_KEYS,
        how='outer',
    )

    if balance is not None and not balance.empty:
        merged = merged.merge(
            # `fp` is already on the left from the flow statements, and the
            # balance sheet's copy of it is not part of the join.
            balance.drop(columns=['fp', 'frame'], errors='ignore'),
            on=PIT_MERGE_KEYS,
            how='left',
        )

    merged = _clean_fiscal_year(merged)
    merged = _fill_from_earlier_filings(merged)
    return _flag_latest(merged)


def _clean_fiscal_year(df: pd.DataFrame) -> pd.DataFrame:
    """Label each period by the year it covers, not the year it was filed in.

    A period reported again in a later filing arrives carrying that filing's
    fiscal year, which would put one period under two labels.
    """
    df['fy'] = df.groupby(
        ['cik', 'fp', 'start', 'end'], dropna=False)['fy'].transform('min')
    return df


def _fill_from_earlier_filings(df: pd.DataFrame) -> pd.DataFrame:
    """Fill gaps in a restatement from the filing that reported the period first.

    A later filing often carries a period as a comparative and repeats only
    some of it. The rest was not withdrawn, it simply was not restated.
    """
    df = df.sort_values('filed')
    group = ['cik', 'fy', 'fp', 'form', 'start', 'end']
    fixed = set(group) | {'accn', 'period', 'filed', 'frame'}
    fill_cols = [c for c in df.columns if c not in fixed]

    df[fill_cols] = df.groupby(group, dropna=False)[fill_cols].ffill()
    return df


def _flag_latest(df: pd.DataFrame) -> pd.DataFrame:
    """Mark the most recently filed version of each period.

    Every version is kept - which one is correct depends on when you are
    asking - so this only labels the newest rather than dropping the rest.
    """
    latest = df.groupby(
        ['cik', 'start', 'end', 'period'], dropna=False)['filed'].transform('max')
    df['is_latest'] = df['filed'].eq(latest)
    return df


# === Helpers === #


def _value_columns(df: pd.DataFrame, statement) -> list:
    return [dp.name for dp in statement.mappings if dp.name in df.columns]


def _months_between(start: pd.Series, end: pd.Series) -> pd.Series:
    """Length of a period in whole months, null if either end is missing."""
    months = ((end.dt.year - start.dt.year) * 12
              + (end.dt.month - start.dt.month)
              + (end.dt.day - start.dt.day) / 30).round(0)
    return months.where(start.notna() & end.notna())
