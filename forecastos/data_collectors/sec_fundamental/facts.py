"""Reading the companyfacts archive, and reshaping it one statement at a time.

The archive holds one JSON member per filer, each a nesting of
taxonomy -> tag -> unit -> [facts]. Two filters run here rather than downstream,
because both cut the row count before a DataFrame is built at all:

  - only tags the schema can reference are kept, ~150 against a median of ~270
    us-gaap tags per company;
  - only 10-K and 10-Q facts are kept, which is also the entity filter. Funds,
    trusts and shells file N-CSR and 10-D, never a 10-K, so they fall out here
    without needing the 1.4 GB submissions archive to identify them.

Members are read out of the zip in place. The central directory indexes every
entry, so one company is a seek and a single deflate stream while the rest stay
compressed - there is no reason to spend 17 GB of disk unpacking the archive.
"""

import json
import zipfile
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import numpy as np
import pandas as pd

# Only these are read; everything else in a filing describes a period we cannot
# place or an entity we are not collecting.
WANTED_FORMS = frozenset({'10-K', '10-Q'})

# `dei` carries EntityCommonStockSharesOutstanding, the fallback share count.
WANTED_TAXONOMIES = ('us-gaap', 'dei')

FACT_COLS = [
    'cik', 'tag', 'start', 'end', 'val',
    'accn', 'fy', 'fp', 'form', 'filed', 'frame',
]

DATE_COLS = ('start', 'end', 'filed')

# What identifies a filing's reported figure for a period. `filed` is in here
# deliberately: the same period is reported again in later filings, sometimes
# restated, and collapsing those would leave only the newest version - the one
# that was not knowable at the time.
FLOW_INDEX = ['cik', 'accn', 'fy', 'fp', 'form', 'start', 'end', 'filed']

# Point-in-time facts have no start; they are measured at `end`.
PIT_INDEX = ['cik', 'accn', 'fy', 'fp', 'form', 'end', 'filed']


def read_facts(
    archive_path,
    tags: frozenset,
    ciks: Optional[list] = None,
    workers: int = 1,
) -> pd.DataFrame:
    """Every fact in the archive that `tags` names, as one long frame.

    Args:
        archive_path: path to companyfacts.zip.
        tags: XBRL tags to keep. Anything else is discarded before it becomes a
            row - this is the single largest reduction in the pipeline.
        ciks: restrict to these filers. None reads all ~18k.
        workers: processes to parse with. Each opens the archive itself, so
            they share no state. 1 keeps it in-process.

    Returns:
        One row per fact: cik, tag, start, end, val, accn, fy, fp, form, filed,
        frame. Dates are datetime64; `cik` is the padded 10-digit string.
    """
    with zipfile.ZipFile(archive_path) as archive:
        names = ([_member_name(cik) for cik in ciks] if ciks
                 else [n for n in archive.namelist() if n.endswith('.json')])

    if workers > 1:
        frames = _read_parallel(archive_path, names, tags, workers)
    else:
        with zipfile.ZipFile(archive_path) as archive:
            frames = []
            for i, name in enumerate(names, 1):
                frames.append(_read_member(archive, name, tags))
                if i % 250 == 0 or i == len(names):
                    print(f'\r  parsed {i} / {len(names)} companies', end='')
    print()

    # Companies that reported none of the wanted tags contribute an empty
    # frame, and concatenating those would let their all-null columns decide
    # the result's dtypes.
    frames = [f for f in frames if not f.empty]
    if not frames:
        return _clean(pd.DataFrame(columns=FACT_COLS))

    return _clean(pd.concat(frames, ignore_index=True))


def pivot_statement(facts: pd.DataFrame, statement) -> pd.DataFrame:
    """One row per filing-period, one column per tag the statement reads.

    Args:
        facts: the long frame from `read_facts`.
        statement: a `StatementSchema`.

    Returns:
        A frame indexed by the statement's key columns, plus one column per tag
        and the `frame` label SEC assigned the period where it assigned one.
    """
    index_cols = PIT_INDEX if statement.is_pit else FLOW_INDEX

    rows = facts[facts['tag'].isin(statement.required_tags)]
    # A fact missing any part of its key cannot be placed on a timeline, and
    # one missing its value has nothing to contribute.
    rows = rows.dropna(subset=index_cols + ['tag', 'val'])
    # The same tag can appear twice under one key when a filing reports it in
    # more than one unit. First wins, as before.
    rows = rows.drop_duplicates(subset=index_cols + ['tag'], keep='first')

    if rows.empty:
        return pd.DataFrame(columns=index_cols + ['frame'])

    wide = rows.pivot(index=index_cols, columns='tag', values='val')
    wide = wide.rename_axis(None, axis=1)

    # `frame` belongs to the period rather than to any one tag, so it is
    # collapsed alongside the pivot instead of becoming a column in it.
    labels = rows.groupby(index_cols, sort=False)['frame'].first()
    wide = wide.join(labels)

    return wide.reset_index()


def _read_parallel(archive_path, names, tags, workers):
    """Parse members across processes, each with its own handle on the archive.

    Parsing is the bulk of the load and every member is independent, so this
    scales close to linearly until the concat at the end.
    """
    chunks = [names[i::workers] for i in range(workers)]
    frames = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_read_chunk, str(archive_path), chunk, tags)
                   for chunk in chunks]
        for i, future in enumerate(futures, 1):
            frames.append(future.result())
            print(f'\r  parsed {i} / {len(futures)} chunks', end='')
    return frames


def _read_chunk(archive_path: str, names: list, tags: frozenset) -> pd.DataFrame:
    """Parse a slice of members. Runs in a worker process."""
    with zipfile.ZipFile(archive_path) as archive:
        frames = [_read_member(archive, name, tags) for name in names]

    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=FACT_COLS)
    return pd.concat(frames, ignore_index=True)


def _read_member(
    archive: zipfile.ZipFile,
    name: str,
    tags: frozenset,
) -> pd.DataFrame:
    """Decompress one company's member and flatten the tags we want out of it."""
    try:
        payload = archive.read(name)
    except KeyError:
        # A filer with no XBRL facts has no member at all, which is a data
        # answer rather than a corrupt archive.
        raise ValueError(
            f'{name} is not in the archive - that filer has no XBRL facts'
        ) from None

    return _to_facts(json.loads(payload), tags, _member_cik(name))


def _member_name(cik) -> str:
    """Members are named by the padded CIK, e.g. 'CIK0000320193.json'."""
    digits = ''.join(c for c in str(cik) if c.isdigit())
    if not digits:
        raise ValueError(f'not a cik: {cik!r}')
    return f'CIK{digits.zfill(10)}.json'


def _member_cik(name: str) -> str:
    """The CIK a member is named for, padded."""
    digits = ''.join(c for c in name if c.isdigit())
    return digits.zfill(10)


def _to_facts(company: dict, tags: frozenset, member_cik: str) -> pd.DataFrame:
    """Flatten one company into a row per fact, keeping only wanted tags.

    The tag and form tests run against the parsed dicts, before any row is
    built. Discarding here rather than after the frame exists is what keeps the
    intermediate roughly a quarter of the size it would otherwise be.
    """
    records, tag_labels, counts = [], [], []

    # A handful of members are an empty object - a filer SEC has an entry for
    # and no facts under. There is nothing to read out of them.
    facts = company.get('facts', {})
    for taxonomy in WANTED_TAXONOMIES:
        for tag, body in facts.get(taxonomy, {}).items():
            if tag not in tags:
                continue
            for unit_facts in body.get('units', {}).values():
                kept = [f for f in unit_facts if f.get('form') in WANTED_FORMS]
                if not kept:
                    continue
                records.extend(kept)
                tag_labels.append(tag)
                counts.append(len(kept))

    if not records:
        return pd.DataFrame(columns=FACT_COLS)

    # The fact dicts go to pandas in one batch and the tag is attached
    # afterwards by repeating it over its own run. Building a row dict per fact
    # would copy every record back through Python a field at a time.
    df = pd.DataFrame(records)
    df['tag'] = np.repeat(np.array(tag_labels, dtype=object), counts)

    # Padded, so a CIK read back from a csv still joins against SEC's own
    # ticker mappings rather than losing its leading zeros to an int cast.
    # About a quarter of members carry the CIK as a string rather than an int,
    # and the member name is authoritative for the rest, so neither the type
    # nor the presence of the field is relied on.
    df['cik'] = member_cik

    # reindex rather than a plain column select: `start` and `frame` are absent
    # from most records and can be missing from a company's facts entirely, and
    # the schema should not change shape because of who was asked for.
    return df.reindex(columns=FACT_COLS)


def _clean(facts: pd.DataFrame) -> pd.DataFrame:
    """Type the columns once, here, rather than per statement downstream."""
    for col in DATE_COLS:
        facts[col] = pd.to_datetime(facts[col], errors='coerce')

    facts['val'] = pd.to_numeric(facts['val'], errors='coerce')

    # A missing fiscal year is not a reason to drop the fact - `start`/`end`
    # carry the period, and `fy` only labels the filing it arrived in.
    facts['fy'] = facts['fy'].fillna(0).astype('int64')

    return facts
