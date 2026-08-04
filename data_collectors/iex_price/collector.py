from pathlib import Path
import pandas as pd
import requests
from ..downloader import FileDownloader, TIMEOUT_SEC
from .tops_trade_scanner import scan_trades

HIST_URL = 'https://iextrading.com/api/1.0/hist'
DEFAULT_DATA_DIR = str(Path.cwd() / 'data')

# Enough to identify and download a capture; the API returns more than this.
CATALOG_COLS = ['date', 'feed', 'version', 'protocol', 'size_bytes', 'link']


class IEXPriceCollector:
    """Daily OHLCV bars built from IEX's public TOPS capture archive.

    Args:
        data_dir: where capture files are downloaded. Defaults to ./data, resolved
            against the working directory as of import.
        cleanup: delete each capture file once it has been parsed. Leave True for
            one-off runs; pass False while iterating, since re-fetching a
            session means another 10+ GB download.
    """

    def __init__(self, data_dir: str = DEFAULT_DATA_DIR, cleanup: bool = True):
        self.downloader = FileDownloader(data_dir, cleanup)

    def collect(self, start: str = None, end: str = None) -> pd.DataFrame:
        """Bars for every session in [start, end]; either bound may be None.

        Raises ValueError if no session falls in the range, so a typo'd date
        fails immediately rather than after a download.
        """
        catalog = self.fetch_tops_catalog(start, end)
        if catalog.empty:
            raise ValueError(f'no TOPS sessions between {start} and {end}')

        # One session at a time: a capture runs to tens of GB, but only the
        # aggregated bars (a few hundred KB) survive each iteration.
        bars = [self.daily_bars(entry) for entry in catalog.to_dict('records')]
        return pd.concat(bars, ignore_index=True)

    def daily_bars(self, entry: dict) -> pd.DataFrame:
        """Download one capture, parse its trades, reduce them to OHLCV bars.

        `entry` is one catalog row as a dict - it carries the download link plus
        the fields that name the local file.
        """
        with self.downloader.fetch(
            entry['link'], _capture_filename(entry)
        ) as path:
            trades = scan_trades(path, entry['date'])

        return _to_daily_bars(trades, entry['date'])

    @staticmethod
    def fetch_tops_catalog(start: str = None, end: str = None) -> pd.DataFrame:
        """Returns available TOPS feeds from IEX between [start, end] (one file per trading day)"""
        resp = requests.get(HIST_URL, timeout=TIMEOUT_SEC)
        resp.raise_for_status()

        # The archive is keyed by session; flatten to one row per capture.
        rows = [entry for entries in resp.json().values() for entry in entries]
        df = pd.DataFrame(rows)

        # `size` arrives as text. Cast before it is used as a sort key below -
        # compared as strings, the 242-byte stub "242" outranks "11509902176".
        df['size_bytes'] = df['size'].astype('int64')
        df = df[df['feed'] == 'TOPS']

        # Either bound may be None, meaning no limit on that side
        dates = pd.to_datetime(df['date'])
        df = df[dates.between(
            pd.Timestamp(start) if start else dates.min(),
            pd.Timestamp(end) if end else dates.max(),
        )]

        # Aug-Nov 2017 published 1.5 and 1.6 for the same session (identical
        # trades, different wire format), and rows for the retired feed can be
        # 242-byte stubs. Sorting by largest size first keeps the real capture
        # even when the newer feed is the stub; version breaks any remaining tie.
        df = (
            df.sort_values(['size_bytes', 'version'], ascending=False)
            .drop_duplicates(subset='date', keep='first')
            .sort_values('date', ascending=False)
            .reset_index(drop=True)
        )

        return df[CATALOG_COLS]


def _capture_filename(entry: dict) -> str:
    """Name captures by session rather than by the opaque tail of the link."""
    return (
        f"{entry['date']}_"
        f"{entry['protocol']}_"
        f"{entry['feed']}{entry['version']}.pcap.gz"
    )


def _to_daily_bars(trades: pd.DataFrame, date: str) -> pd.DataFrame:
    """Collapse one session's trades into a bar per symbol.

    `first`/`last` are positional, so the sort is what makes open and close the
    day's first and final prints rather than whichever row pandas saw first.
    """
    trades = trades.sort_values(['symbol', 'ts', 'trade_id'])

    bars = (
        # sort=False: already ordered by symbol, so skip re-sorting the groups
        trades.groupby('symbol', sort=False)
        .agg(
            open=('price', 'first'),
            high=('price', 'max'),
            low=('price', 'min'),
            close=('price', 'last'),
            volume=('size', 'sum')
        )
        .reset_index()
    )

    bars.insert(0, 'date', date)
    return bars
