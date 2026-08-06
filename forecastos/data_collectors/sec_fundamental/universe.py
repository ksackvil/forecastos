"""Which CIK traded under which symbol, as of today.

SEC publishes the mapping as current state only. A company that has been
acquired or delisted keeps its filing history and its CIK, but its ticker and
exchange are blanked - Twitter's record still holds every filing through 2022
and an empty `tickers` list. Nothing SEC publishes says what a company used to
trade as.

So this is a snapshot, and a snapshot is only useful to a historical dataset if
it is kept. Collect it alongside each run and append; `merge_snapshot` folds a
new one into the accumulated history, closing rows that changed and leaving the
rest alone. What that cannot do is recover symbols from before the first run -
for those, a vendor with its own history is the only source.
"""

import pandas as pd

TICKERS_URL = 'https://www.sec.gov/files/company_tickers_exchange.json'

TICKERS_FILENAME = 'company_tickers_exchange.json'

UNIVERSE_COLS = ['cik', 'name', 'ticker', 'exchange']

# Open-ended validity, so a row that is still current sorts and compares
# without needing null handling everywhere.
FOREVER = pd.Timestamp('2262-04-11')


def to_universe(payload: dict, as_of: pd.Timestamp) -> pd.DataFrame:
    """One row per listing in SEC's ticker file.

    Args:
        payload: the parsed company_tickers_exchange.json.
        as_of: the date this snapshot describes.

    Returns:
        cik, name, ticker, exchange, as_of. A CIK appears once per symbol, so
        companies with more than one share class have more than one row.
    """
    df = pd.DataFrame(payload['data'], columns=payload['fields'])
    df = df[UNIVERSE_COLS].copy()

    # Padded, to join against CIKs read back from a csv without losing leading
    # zeros to an int cast.
    df['cik'] = df['cik'].astype('int64').map('{:010d}'.format)
    df['as_of'] = as_of
    return df


def merge_snapshot(
    history: pd.DataFrame,
    snapshot: pd.DataFrame,
) -> pd.DataFrame:
    """Fold a snapshot into accumulated history, as validity ranges.

    Rows are closed rather than deleted, so a symbol a company no longer holds
    stays queryable for the period it did hold it. That is the whole point of
    keeping the history: joining fundamentals to a symbol is only correct if
    the join carries the date the symbol was in use.

    Args:
        history: previous output of this function, or an empty frame on the
            first run.
        snapshot: output of `to_universe`.

    Returns:
        cik, ticker, exchange, name, valid_from, valid_to. `valid_to` is
        open-ended for listings still current as of the snapshot.
    """
    as_of = snapshot['as_of'].iloc[0]
    incoming = snapshot.drop(columns=['as_of']).assign(
        valid_from=as_of, valid_to=FOREVER)

    if history is None or history.empty:
        return incoming.reset_index(drop=True)

    keys = ['cik', 'ticker', 'exchange']
    open_rows = history['valid_to'].eq(FOREVER)

    current = history.loc[open_rows]
    still_listed = current[keys].apply(tuple, axis=1).isin(
        incoming[keys].apply(tuple, axis=1))

    # A listing that has dropped out of the file ended some time between the
    # last snapshot and this one; the last day we saw it is the honest bound.
    history = history.copy()
    closed = current.index[~still_listed]
    history.loc[closed, 'valid_to'] = as_of

    known = current.loc[still_listed, keys].apply(tuple, axis=1)
    fresh = incoming[~incoming[keys].apply(tuple, axis=1).isin(known)]

    return pd.concat([history, fresh], ignore_index=True)
