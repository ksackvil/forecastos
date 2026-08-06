"""Reading the companyfacts archive into one fact per row.

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


def read_facts(
    archive_path,
    tags: frozenset,
    ciks: Optional[list] = None,
) -> pd.DataFrame:
    """Every fact in the archive that `tags` names, as one long frame.

    Args:
        archive_path: path to companyfacts.zip.
        tags: XBRL tags to keep. Anything else is discarded before it becomes a
            row - this is the single largest reduction in the pipeline.
        ciks: restrict to these filers. None reads all ~19k.

    Returns:
        One row per fact: cik, tag, start, end, val, accn, fy, fp, form, filed,
        frame. Dates are datetime64; `cik` is the padded 10-digit string.
    """
    with zipfile.ZipFile(archive_path) as archive:
        names = ([_member_name(cik) for cik in ciks] if ciks
                 else [n for n in archive.namelist() if n.endswith('.json')])

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
