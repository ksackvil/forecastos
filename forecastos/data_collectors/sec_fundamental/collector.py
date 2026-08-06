"""Fundamental statements built from SEC's bulk XBRL companyfacts archive."""

import os
from pathlib import Path

import pandas as pd

from ..downloader import FileDownloader
from .extract import extract
from .facts import read_facts
from .normalize import merge_statements, normalize
from .schema import Schema

DEFAULT_DATA_DIR = str(Path.cwd() / 'data')

ARCHIVE_FILENAME = 'companyfacts.zip'

COMPANY_FACTS_URL = 'https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip'

SCHEMA_DIR = os.path.join(os.path.dirname(__file__), 'schemas')


class SECFundamentalCollector:
    """Income, cash flow and balance sheet figures for every SEC XBRL filer.

    Statements are keyed by CIK and by two dates: the period a figure covers,
    and the filing it arrived in. Nothing here is keyed by ticker. A ticker is
    a label a company holds for a while - it moves between share classes, gets
    reused after a delisting, and SEC blanks it the moment a company is taken
    over - so joining fundamentals to it would quietly mis-key exactly the
    companies that make a backtest honest. Symbols belong in a separate
    mapping, joined on CIK and bounded by the dates the symbol was in use.

    Args:
        user_agent: sent to SEC, which rejects unidentified clients. They ask
            for contact details, e.g. 'Example Corp info@example.com'.
        data_dir: where the archive is downloaded. Defaults to ./data, resolved
            against the working directory as of import.
        cleanup: whether to delete the archive once collection finishes. It is
            1.2 GB and SEC rebuilds it nightly, so keeping it is only worth it
            across runs on the same day.
    """

    def __init__(
        self,
        user_agent: str,
        data_dir: str = DEFAULT_DATA_DIR,
        cleanup: bool = True,
    ):
        self.sec_request_headers = {'User-Agent': user_agent}
        self.downloader = FileDownloader(
            data_dir, cleanup, self.sec_request_headers)
        self.schema = Schema.from_dir(SCHEMA_DIR)

    def collect(self, ciks: list = None) -> pd.DataFrame:
        """Fundamental statements, one row per company per period per filing.

        The whole archive takes a few minutes. To go wider, shard `ciks` across
        separate jobs rather than reaching for threads or processes in here -
        every stage shards cleanly by company, where parallelising the read
        alone would only touch the smaller half of the work.

        Args:
            ciks: CIKs in any form that carries the digits - 320193,
                '0000320193' and 'CIK0000320193' all name Apple. None collects
                every filer in the archive, which is what you want for anything
                historical: companies that have since been acquired or delisted
                are in here with their filing history intact, and no later run
                can recover them once they have been left out.

        Returns:
            One row per company-period-filing: cik, accn, fy, fp, form, start,
            end, period, filed, frame, is_latest, and one column per datapoint.
            Balance sheet figures arrive as `start_`/`end_` pairs, since they
            are measured at an instant rather than over the period.

            `cik` is the padded 10-digit string SEC keys filers by. `start`,
            `end` and `filed` are datetimes; `period` is the period's length in
            months. `frame` is SEC's own calendar label, set only where SEC
            could fit the period to a calendar quarter.

            The same period appears once per filing that reported it, so
            restatements sit alongside the original rather than replacing it.
            Filter to `filed <= your date` to see a period as it was known
            then, or to `is_latest` for the most recent version of each.

        Raises:
            ValueError: a named CIK has no member in the archive.
        """
        statements = self.collect_statements(ciks=ciks)
        return merge_statements(statements)

    def collect_statements(self, ciks: list = None) -> dict:
        """The statements `collect` merges, before they are joined together.

        Useful when a statement is wanted on its own, or to see figures that
        the merge drops - a balance sheet whose period has no income statement
        has no row to attach to.

        Args:
            ciks: as `collect`.

        Returns:
            Normalized frames keyed by statement name: income_statement,
            cashflow_statement, balance_sheet, other.
        """
        with self.downloader.fetch(COMPANY_FACTS_URL, ARCHIVE_FILENAME) as path:
            print('reading facts')
            facts = read_facts(path, self.schema.required_tags, ciks=ciks)

        statements = {}
        for statement in self.schema:
            print(f'building {statement.name}')
            statements[statement.name] = normalize(
                extract(facts, statement), statement)

        return statements
