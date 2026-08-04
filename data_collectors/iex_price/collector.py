from pathlib import Path
import pandas as pd
import requests
from ..downloader import FileDownloader
from .tops_trade_scanner import scan_trades

HIST_URL = 'https://iextrading.com/api/1.0/hist'

DEFAULT_DATA_DIR = Path.cwd() / 'data'


class IEXPriceCollector():
    def __init__(
        self,
        data_dir: str | Path = DEFAULT_DATA_DIR,
        cleanup: bool = True
    ):
        self.downloader = FileDownloader(data_dir, cleanup)

    def collect(
        self,
        start: str | None = None,
        end: str | None = None,
        progress_every: int | None = 10
    ) -> pd.DataFrame:
        catalog = self.fetch_tops_catalog(start, end)
        if catalog.empty:
            raise ValueError(f'no TOPS sessions between {start} and {end}')

        bars = [
            self.daily_bars(entry, progress_every)
            for entry in catalog.to_dict('records')
        ]

        return pd.concat(bars, ignore_index=True)

    def daily_bars(
        self,
        entry: dict,
        progress_every: int | None = 10
    ) -> pd.DataFrame:
        """Download one capture, parse its trades, reduce them to OHLCV bars."""
        with self.downloader.fetch(
            entry['link'], _capture_filename(entry)
        ) as path:
            trades = scan_trades(
                path,
                date=entry['date'],
                progress_every=progress_every,
            )

        return _to_daily_bars(trades, entry['date'])

    @staticmethod
    def fetch_tops_catalog(
        start: str | None = None,
        end: str | None = None,
        timeout_sec: int = 30
    ) -> pd.DataFrame:
        resp = requests.get(HIST_URL, timeout=timeout_sec)
        resp.raise_for_status()

        rows = [entry for entries in resp.json().values() for entry in entries]
        df = pd.DataFrame(rows)
        df['size_bytes'] = df['size'].astype('int64')

        # Keep only user TOPS feeds
        df = df[df["feed"] == "TOPS"]

        # Keep only sessions in range; either bound may be None
        dates = pd.to_datetime(df['date'])
        df = df[dates.between(
            pd.Timestamp(start) if start else dates.min(),
            pd.Timestamp(end) if end else dates.max(),
        )]

        # One capture per session
        # Aug-Nov 2017 published 1.5 and 1.6 for the same session (identical trades,
        # different wire format), and rows for the retired feed can be 242-byte stubs.
        # Sorting by size first keeps the real capture even when the newer feed is the
        # stub; version breaks any remaining tie.
        df = (
            df.sort_values(['size_bytes', 'version'], ascending=False)
            .drop_duplicates(subset='date', keep='first')
            .sort_values('date', ascending=False)
            .reset_index(drop=True)
        )

        return df[[
            "date",
            "feed",
            "version",
            "protocol",
            "size_bytes",
            "link"
        ]]


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
    trades = trades.sort_values(["symbol", "ts", "trade_id"])

    bars = (
        trades.groupby("symbol")
        .agg(
            open=("price", "first"),
            high=("price", "max"),
            low=("price", "min"),
            close=("price", "last"),
            volume=("size", "sum")
        )
        .reset_index()
    )

    bars.insert(0, "date", date)
    return bars
