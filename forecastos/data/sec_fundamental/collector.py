"""Fundamental statements built from SEC's bulk XBRL companyfacts archive."""

from pathlib import Path

import pandas as pd

from ..downloader import FileDownloader
from .extract import extract
from .facts import read_facts
from .normalize import merge_statements, normalize
from .schema import load_schema, required_tags

DEFAULT_DATA_DIR = str(Path.cwd() / 'data')

ARCHIVE_FILENAME = 'companyfacts.zip'

COMPANY_FACTS_URL = 'https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip'

SCHEMA_DIR = Path(__file__).parent / 'schemas'


class SECFundamental:
    """Income, cash flow and balance sheet figures for every SEC XBRL filer.

    Keyed by CIK and by two dates: the period covered, and the filing it
    arrived in. Never by ticker - tickers move between share classes, get
    reused after a delisting, and are blanked on takeover, so they would
    mis-key exactly the companies that make a backtest honest.

    Args:
        user_agent: sent to SEC, which rejects unidentified clients. Contact
            details, e.g. 'Example Corp info@example.com'.
        data_dir: where the archive is downloaded. Defaults to ./data.
        cleanup: delete the 1.2 GB archive when done. SEC rebuilds it nightly,
            so keeping it only helps across runs on the same day.
    """

    def __init__(
        self,
        user_agent: str,
        data_dir: str = DEFAULT_DATA_DIR,
        cleanup: bool = True,
    ):
        self.downloader = FileDownloader(
            data_dir, cleanup, {'User-Agent': user_agent})
        self.schema = load_schema(SCHEMA_DIR)

    def collect(self, ciks: list = None) -> pd.DataFrame:
        """Fundamental statements, one row per company per period per filing.

        The whole archive takes a few minutes. To go wider, shard `ciks` across
        separate jobs - every stage shards cleanly by company.

        Args:
            ciks: any form carrying the digits - 320193, '0000320193' and
                'CIK0000320193' all name Apple. None collects every filer,
                including companies since acquired or delisted, which is what
                you want for anything historical.

        Returns:
            One row per company-period-filing: cik, accn, fy, fp, form, start,
            end, period, filed, frame, is_latest, and one column per datapoint.
            Balance sheet figures come as `start_`/`end_` pairs, being instants
            rather than spans.

            `cik` is the padded 10-digit string. `start`, `end` and `filed` are
            datetimes; `period` is the period's length in months. `frame` is
            SEC's calendar label, set only where the period fit a quarter.

            Restatements sit alongside the original. Filter to
            `filed <= your date` for a point-in-time view, or to `is_latest`
            for the newest version of each period.

        Raises:
            ValueError: a named CIK has no member in the archive.
        """
        statements = self.collect_statements(ciks=ciks)
        return merge_statements(statements)

    def collect_statements(self, ciks: list = None) -> dict:
        """The statements `collect` merges, before they are joined.

        Useful for one statement on its own, or to see figures the merge drops
        - a balance sheet whose period has no income statement has no row to
        attach to.

        Args:
            ciks: as `collect`.

        Returns:
            Normalized frames keyed by statement name: income_statement,
            cashflow_statement, balance_sheet, other.
        """
        with self.downloader.fetch(COMPANY_FACTS_URL, ARCHIVE_FILENAME) as path:
            print('reading facts')
            facts = read_facts(path, required_tags(self.schema), ciks=ciks)

        statements = {}
        for statement in self.schema:
            print(f'building {statement.name}')
            statements[statement.name] = normalize(
                extract(facts, statement), statement)

        return statements
