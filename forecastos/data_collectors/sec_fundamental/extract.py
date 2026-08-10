"""Turning reported facts into datapoint columns.

Facts arrive one per row. Pivoting them gives a row per filing-period with a
column per tag - the shape datapoints are defined over - so every datapoint is
numpy over whole columns rather than a Python call per row per datapoint, which
with ~45 datapoints over millions of rows dominated everything else.

The tag columns live only between the pivot and the evaluation; nothing
downstream reads them.
"""

import numpy as np
import pandas as pd

# A mostly-empty row is rarely a real statement - usually one figure disclosed
# for a period the rest of the filing does not cover.
MAX_NULL_SHARE = 0.5


def extract(facts: pd.DataFrame, statement) -> pd.DataFrame:
    """One row per filing-period, with a column per datapoint.

    Args:
        facts: the long frame from `read_facts`.
        statement: a `StatementSchema`.

    Returns:
        The statement's key columns, SEC's `frame` label where it assigned one,
        and a column per datapoint. Rows more than half null are dropped.
    """
    names = statement.datapoint_names
    wide = _pivot(facts, statement)
    if wide.empty:
        return wide.reindex(columns=list(wide.columns) + names)

    for datapoint in statement.mappings:
        wide[datapoint.name] = _datapoint_values(wide, datapoint.alternatives)

    _apply_overrides(wide, statement)

    # Calculations derive a datapoint from other datapoint columns and only
    # fill where the mapped value was null. Declaration order matters - one can
    # read a column another writes.
    for datapoint in statement.calculations:
        derived = pd.Series(
            _datapoint_values(wide, datapoint.alternatives), index=wide.index)
        wide[datapoint.name] = (wide[datapoint.name].fillna(derived)
                                if datapoint.name in wide.columns else derived)

    return wide[wide[names].isna().mean(axis=1) <= MAX_NULL_SHARE]


def _pivot(facts: pd.DataFrame, statement) -> pd.DataFrame:
    """Facts for one statement, reshaped to a row per filing-period."""
    keys = statement.key_columns

    rows = facts[facts['tag'].isin(statement.required_tags)]
    # A fact missing part of its key cannot be placed on a timeline.
    rows = rows.dropna(subset=keys + ['tag', 'val'])
    # One tag can appear twice under a key if reported in two units. First wins.
    rows = rows.drop_duplicates(subset=keys + ['tag'], keep='first')

    if rows.empty:
        return pd.DataFrame(columns=keys + ['frame'])

    return (rows.pivot(index=keys, columns='tag', values='val')
            .rename_axis(None, axis=1)
            # `frame` belongs to the period, not to any one tag, so it is
            # collapsed alongside the pivot instead of becoming a column in it.
            .join(rows.groupby(keys, sort=False)['frame'].first())
            .reset_index())


def _apply_overrides(wide: pd.DataFrame, statement) -> None:
    """Recompute the few CIKs whose filings need their own tag order.

    Each override already carries the shared alternatives as a fallback (see
    `_compile_statement`). Only the overridden company's rows are touched - a
    few thousand out of millions, cheap enough to redo rather than thread
    through the pass above.
    """
    for cik, datapoints in statement.overrides.items():
        mask = (wide['cik'] == cik).to_numpy()
        rows = wide.loc[mask]
        for datapoint in datapoints:
            wide.loc[mask, datapoint.name] = _datapoint_values(
                rows, datapoint.alternatives)


def _datapoint_values(frame: pd.DataFrame, alternatives) -> np.ndarray:
    """First alternative that yields a value, per row."""
    out = np.full(len(frame), np.nan)
    for alternative in alternatives:
        missing = np.isnan(out)
        # Every row filled - later alternatives have nothing left to do.
        if not missing.any():
            break
        out = np.where(missing, _alternative_values(frame, alternative), out)
    return out


def _alternative_values(frame: pd.DataFrame, alternative) -> np.ndarray:
    """Sum of the alternative's terms, or null if the sum cannot stand."""
    terms = np.vstack([_term_values(frame, t) for t in alternative.terms])

    if not alternative.allow_null_components:
        # Every component must be present - no partial sums, which is what
        # summing straight through gives us.
        return terms.sum(axis=0)

    # An empty term contributes nothing; the sum fails only if all are.
    return np.where(np.isnan(terms).all(axis=0), np.nan, np.nansum(terms, axis=0))


def _term_values(frame: pd.DataFrame, tags) -> np.ndarray:
    """First of `tags` that the filing reported, per row."""
    out = np.full(len(frame), np.nan)

    for tag in tags:
        if tag.name in frame.columns:
            col = frame[tag.name].to_numpy(dtype='float64') * tag.multiplier
        else:
            # A company never using this tag is ordinary - that is what the
            # next tag in the group is for.
            col = np.full(len(frame), np.nan)

        if tag.default is not None:
            col = np.where(np.isnan(col), tag.default, col)

        if tag.ignore_if_zero:
            # A zero here means the company files the tag but folds the real
            # figure into another, so it should not end the search.
            col = np.where(col == 0, np.nan, col)

        out = np.where(np.isnan(out), col, out)

    return out
