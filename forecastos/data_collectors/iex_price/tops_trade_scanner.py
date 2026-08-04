"""Extract trades from an IEX TOPS capture.

A capture is a gzipped pcap of a whole trading day on the feed - quotes, system
messages and trades interleaved, over 10 GB for a busy capture. Trades are
a small fraction of it.

The brute force approach is to read message by message: read each packet
header, step through the messages inside, keep the trade reports. That spends
almost all its time decoding messages it then discards (~15 min on a ~14 GB capture).

This module instead treats the decompressed stream as bytes. Every trade report
begins with the same three-byte signature, so it scans for that literal, copies
only the matching records, and reinterprets the concatenated result as a numpy
record array in a single pass (~60 sec on a ~10 GB capture).

The tradeoff: packet headers are skipped, so IEX-TP `seq` is unavailable. The
`trade_id` inside each message is monotonic, so it orders trades just as well.
"""

import gzip
import re
import time
from pathlib import Path
import numpy as np
import pandas as pd


# --- TOPS wire format -------------------------------------------------------
# Prices are fixed point with 4 implied decimals.
PRICE_SCALE = 10_000

# A trade message is length-prefixed, so "<length><type>" is a far more selective
# signature than the bare type byte 0x54 — which occurs constantly inside prices,
# sizes and timestamps. Trade Report is 38 bytes on TOPS 1.6, 42 on 1.5.
TRADE_SIGS = (b'\x26\x00T', b'\x2a\x00T')

# Only the first 38 bytes of each match are copied, so 1.5's four trailing bytes
# fall away and both versions share this one dtype.
TRADE_REC = np.dtype({
    'names':    ['flags', 'ts', 'sym', 'size', 'price', 'trade_id'],
    'formats':  ['u1', '<i8', 'S8', '<u4', '<i8', '<i8'],
    'offsets':  [1, 2, 10, 18, 22, 30],
    'itemsize': 38,
})

LEN_PREFIX = 2                  # bytes of "<length>" ahead of the message body
REC_SPAN = LEN_PREFIX + TRADE_REC.itemsize

# --- scan tuning ------------------------------------------------------------
SCAN_CHUNK = 8 << 20
SCAN_TAIL = 64  # > longest trade message + prefix, so none is lost at a chunk edge

SNIFF_STEP = 8 << 20     # grow the probe a chunk at a time...
SNIFF_MAX = 64 << 20     # ...but give up rather than decompress the whole file
SNIFF_MIN_HITS = 64      # enough to separate a real feed from stray false positives

# Timestamps are wall-clock trade times, so they sit within a day of the session
# date. Two days of slack absorbs timezone and pre/post-market spread.
SESSION_SLACK_NS = 2 * 86_400 * 1_000_000_000


def scan_trades(path: Path, date: str, progress_every: int = 10) -> pd.DataFrame:
    """Returns every trade in the capture at `path`, one row each.

    Runs in three steps: work out which trade signature this file uses,
    extract every record that matches it, then reinterpret those bytes as trades.

    `date` is the session the capture covers, and only bounds the timestamp
    plausibility check in `_decode`.
    """
    raw = _carve(path, _sniff_sig(path), progress_every)
    return _decode(raw, *_session_bounds(date))


def _sniff_sig(path: Path) -> bytes:
    r"""Decide which trade signatures this capture uses.

    Worth a separate pass because scanning for both at once costs ~6x: the
    alternation `(?:\x26|\x2a)\x00T` defeats CPython's literal-prefix
    optimization, so `re` runs a general match at every byte rather than a
    memchr scan. One literal keeps `_carve` near the decompression floor.

    Counting hits rather than trusting the catalog's version column also makes a
    capture containing no trades fail here, loudly, instead of quietly producing
    an empty frame. The losing signature scores zero in practice, so this is a
    measurement, not a guess.

    Only reads far enough for one signature to clear SNIFF_MIN_HITS, which a
    normal session does within the first chunk.
    """
    probe = b''
    with gzip.open(path, 'rb') as f:
        while len(probe) < SNIFF_MAX:
            chunk = f.read(SNIFF_STEP)
            if not chunk:
                break
            probe += chunk
            sig = max(TRADE_SIGS, key=probe.count)
            if probe.count(sig) >= SNIFF_MIN_HITS:
                return sig
    scanned = (f'{len(probe) / 1e6:.0f} MB' if len(probe) >= 1e6
               else f'{len(probe):,} bytes')
    raise ValueError(
        f'no TOPS trade messages in first {scanned} of {path}')


def _carve(path: Path, sig: bytes, progress_every: int) -> bytes:
    """Stream the capture and concatenate the body of every signature match.

    Returns one flat bytestring of fixed-width records, ready for `np.frombuffer`
    to reinterpret without copying again.
    """
    pattern = re.compile(re.escape(sig))
    total = Path(path).stat().st_size
    blobs, carry = [], b''
    started = time.time()

    with gzip.open(path, 'rb') as f:
        chunk_no = 0
        while True:
            block = f.read(SCAN_CHUNK)
            last = not block
            buf = carry + block

            # a match starting inside the final SCAN_TAIL bytes may be truncated;
            # defer it to the next round, where `carry` presents it in full
            limit = len(buf) if last else max(0, len(buf) - SCAN_TAIL)

            # skip the length prefix and take exactly TRADE_REC.itemsize bytes,
            # so every record is the same width whatever the TOPS version
            records = [buf[o + LEN_PREFIX:o + REC_SPAN]
                       for o in (m.start() for m in pattern.finditer(buf))
                       if o < limit and o + REC_SPAN <= len(buf)]
            if records:
                blobs.append(b''.join(records))

            if last:
                break
            carry = buf[limit:]

            chunk_no += 1
            if progress_every and chunk_no % progress_every == 0:
                _print_progress(f.fileobj.tell(), total, started)

    return b''.join(blobs)


def _session_bounds(date: str) -> tuple[int, int]:
    """Plausible timestamp window, in ns since the epoch, for session `date`."""
    midnight = pd.Timestamp(date).normalize()
    if midnight.tz is not None:
        midnight = midnight.tz_convert(None)
    return midnight.value - SESSION_SLACK_NS, midnight.value + SESSION_SLACK_NS


def _decode(raw: bytes, ts_lo: int, ts_hi: int) -> pd.DataFrame:
    """Reinterpret carved bytes as records, drop false positives, build the frame."""
    trades = np.frombuffer(raw, dtype=TRADE_REC)

    # ~30 false positives per million hits survive the byte signature; reject them
    # on field plausibility. Deliberately no symbol-charset test — constraining it
    # silently dropped every "GSAH=" trade, and `=` `^` `#` `*` are all legal.
    trades = trades[(trades['size'] > 0) & (trades['price'] > 0)
                    & (trades['ts'] > ts_lo) & (trades['ts'] < ts_hi)]

    # ns since the POSIX epoch on the wire; surfaced as tz-aware UTC so callers
    # need not rediscover the unit and epoch. int64 -> datetime64 is exact.
    return pd.DataFrame({
        'symbol': np.char.strip(trades['sym']).astype(str),
        'ts': pd.to_datetime(trades['ts'].astype('int64'), unit='ns', utc=True),
        'size': trades['size'].astype('int64'),
        'price': trades['price'] / PRICE_SCALE,
        'trade_id': trades['trade_id'].astype('int64'),
    })


def _print_progress(done: int, total: int, started: float) -> None:
    elapsed = time.time() - started
    eta = elapsed * (total / done - 1) if done else 0
    print(f"\r  {done / 1e9:5.2f}/{total / 1e9:.2f} GB ({done / total:4.0%})"
          f"  {elapsed:5.0f}s elapsed, ~{eta:.0f}s left", end="")
