"""Turning tag columns into datapoint columns.

Every datapoint resolves the same way - first non-null alternative, where an
alternative sums groups and a group takes the first tag the filing reported -
and none of that depends on which row is being looked at. So the whole thing is
expressible as numpy over whole columns.

That matters because the row-at-a-time form costs one Python call per row per
datapoint: with ~45 datapoints over millions of rows it dominates everything
else the pipeline does. Evaluating a column at a time makes the cost
proportional to the number of tags instead, and the row count drops out.
"""

import numpy as np
import pandas as pd

# A row where most datapoints came back empty is almost never a real statement.
# It is usually a single figure disclosed for a period the rest of the filing
# does not cover, and carrying it forward would put a lone number on a row that
# reads like a full period.
MAX_NULL_SHARE = 0.5


def extract(wide: pd.DataFrame, statement) -> pd.DataFrame:
    """Add one column per datapoint to a pivoted statement.

    Args:
        wide: output of `pivot_statement`, one column per XBRL tag.
        statement: a `StatementSchema`.

    Returns:
        `wide` with a column per datapoint added, and rows dropped where more
        than half of those columns came back null.
    """
    if wide.empty:
        return wide.reindex(columns=list(wide.columns) + _all_names(statement))

    n = len(wide)

    for datapoint in statement.mappings:
        wide[datapoint.name] = _datapoint_values(
            wide, datapoint.alternatives, n)

    _apply_overrides(wide, statement)

    # Calculations are a fallback layer: they derive a datapoint from other
    # datapoint columns, and only fill in where the mapped value was null. They
    # run in declaration order because one can read a column another writes.
    for datapoint in statement.calculations:
        if datapoint.name not in wide.columns:
            wide[datapoint.name] = np.nan
        mapped = wide[datapoint.name].to_numpy(dtype='float64', copy=True)
        derived = _datapoint_values(wide, datapoint.alternatives, n)
        wide[datapoint.name] = np.where(np.isnan(mapped), derived, mapped)

    return _drop_sparse_rows(wide, statement)


def _apply_overrides(wide: pd.DataFrame, statement) -> None:
    """Recompute a handful of CIKs whose filings need their own tag order.

    Overrides are tried ahead of the shared mappings rather than replacing
    them, so the datapoint still falls back to the usual tags. Only the
    overridden company's rows are touched, which is a few thousand out of
    millions - cheap enough to redo rather than thread through the vectorized
    path above.
    """
    for cik, datapoints in statement.overrides.items():
        mask = (wide['cik'] == cik).to_numpy()
        if not mask.any():
            continue

        rows = wide.loc[mask]
        for datapoint in datapoints:
            alternatives = statement.alternatives_for(datapoint.name, cik)
            wide.loc[mask, datapoint.name] = _datapoint_values(
                rows, alternatives, len(rows))


def _datapoint_values(frame: pd.DataFrame, alternatives, n: int) -> np.ndarray:
    """First alternative that yields a value, per row."""
    out = np.full(n, np.nan)
    for alternative in alternatives:
        # Once every row has a value there is nothing left for a later
        # alternative to fill, and the remaining tags need not be touched.
        if not np.isnan(out).any():
            break
        out = np.where(np.isnan(out), _alternative_values(
            frame, alternative, n), out)
    return out


def _alternative_values(frame: pd.DataFrame, alternative, n: int) -> np.ndarray:
    """Sum of the alternative's groups, or null if the sum cannot stand."""
    groups = np.vstack([_group_values(frame, g, n)
                        for g in alternative.groups])
    missing = np.isnan(groups)

    if alternative.allow_null_components:
        # A group that found nothing contributes nothing; the sum only fails
        # when no group found anything at all.
        failed = missing.all(axis=0)
    else:
        # Every component must be present, so a partial sum is not reported.
        failed = missing.any(axis=0)

    return np.where(failed, np.nan, np.nansum(groups, axis=0))


def _group_values(frame: pd.DataFrame, group, n: int) -> np.ndarray:
    """First tag in the group that the filing reported, per row."""
    out = np.full(n, np.nan)

    for tag in group.tags:
        if tag.name in frame.columns:
            col = frame[tag.name].to_numpy(dtype='float64', copy=True)
            col *= tag.multiplier
        else:
            # A company that never used this tag is the ordinary case, not an
            # error - that is what the next tag in the group is for.
            col = np.full(n, np.nan)

        if tag.default is not None:
            col = np.where(np.isnan(col), tag.default, col)

        if tag.ignore_if_zero:
            # A reported zero here means the company files the tag but folds
            # the real figure into another one, so it should not stop the
            # search the way a genuine value would.
            col = np.where(col == 0, np.nan, col)

        out = np.where(np.isnan(out), col, out)

    return out


def _drop_sparse_rows(wide: pd.DataFrame, statement) -> pd.DataFrame:
    """Drop rows where more than half the datapoints came back null."""
    names = [c for c in _all_names(statement) if c in wide.columns]
    if not names:
        return wide

    null_share = wide[names].isna().sum(axis=1) / len(names)
    return wide[null_share <= MAX_NULL_SHARE]


def _all_names(statement) -> list:
    """Every datapoint column, mappings then calculations, without repeats."""
    names = [dp.name for dp in statement.mappings]
    names += [dp.name for dp in statement.calculations if dp.name not in names]
    return names
