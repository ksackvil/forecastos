"""Smoke tests for the SEC fundamental collector.

Runs the pipeline end to end against one real companyfacts member rather than a
synthetic frame, so the shapes SEC actually ships - cumulative cash flows,
instant balance sheet facts, periods that need a quarter implied - are the ones
under test.

The fixture is Apple, trimmed to the tags the schema reads: 10-K/10-Q facts
from 2020 on, plus every fact on a form the reader is meant to drop, so the
form filter has something to exclude. To refresh it:

    curl -H 'User-Agent: <name> <you@example.com>' \
      https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json

then filter it the same way - the full file is ~3.8 MB.
"""

import json
import zipfile
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pytest

from forecastos.data_collectors.sec_fundamental import SECFundamentalCollector
from forecastos.data_collectors.sec_fundamental.collector import SCHEMA_DIR
from forecastos.data_collectors.sec_fundamental.extract import extract
from forecastos.data_collectors.sec_fundamental.facts import read_facts
from forecastos.data_collectors.sec_fundamental.schema import (
    _compile_sum,
    load_schema,
    required_tags,
)

CIK = '0000320193'
FIXTURE = Path(__file__).parent / f'CIK{CIK}.json'


@pytest.fixture(scope='module')
def schema():
    return load_schema(SCHEMA_DIR)


@pytest.fixture(scope='module')
def archive(tmp_path_factory):
    """The fixture packed the way SEC ships companyfacts.zip."""
    path = tmp_path_factory.mktemp('sec') / 'companyfacts.zip'
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(FIXTURE, FIXTURE.name)
    return path


@pytest.fixture(scope='module')
def facts(archive, schema):
    return read_facts(archive, required_tags(schema), ciks=[CIK])


@pytest.fixture(scope='module')
def collected(archive):
    """`collect()` over the fixture, with the 1.2 GB download stubbed out."""
    collector = SECFundamentalCollector(
        'forecastos smoke-test@example.com', data_dir=str(archive.parent))

    @contextmanager
    def fetch(url, filename):
        yield archive

    collector.downloader.fetch = fetch
    return collector.collect(ciks=[CIK])


def test_schema_compiles(schema):
    by_name = {s.name: s for s in schema}
    assert set(by_name) == {'income_statement', 'cashflow_statement',
                            'balance_sheet', 'other'}

    income = by_name['income_statement']
    assert 'revenue' in income.datapoint_names
    # a calculation-only fallback is still an output column
    assert 'gross_revenue' in income.datapoint_names
    assert 'total_assets' in by_name['balance_sheet'].datapoint_names

    # balance sheets are instants, so they carry no `start`
    assert by_name['balance_sheet'].is_pit
    assert 'start' not in by_name['balance_sheet'].key_columns
    assert 'start' in income.key_columns

    assert len(required_tags(schema)) > 100


def test_sum_rejects_a_bare_string():
    """Without the guard this compiles one Tag per character, silently."""
    with pytest.raises(ValueError):
        _compile_sum({'sum': 'Assets'})


def test_read_facts(facts):
    assert not facts.empty
    assert set(facts['cik']) == {CIK}          # padded to 10 digits
    assert facts['end'].notna().all()

    assert pd.api.types.is_datetime64_any_dtype(facts['end'])
    assert pd.api.types.is_datetime64_any_dtype(facts['filed'])
    assert pd.api.types.is_numeric_dtype(facts['val'])


def test_read_facts_keeps_only_annual_and_quarterly_reports(facts):
    """8-K and 10-K/A describe periods the pipeline cannot place."""
    shipped = {fact['form']
               for tags in json.loads(FIXTURE.read_text())['facts'].values()
               for body in tags.values()
               for unit in body['units'].values()
               for fact in unit}
    # the fixture has to carry excluded forms for this to prove anything
    assert {'8-K', '10-K/A'} <= shipped

    assert set(facts['form']) == {'10-K', '10-Q'}


def test_read_facts_rejects_an_absent_filer(archive, schema):
    with pytest.raises(ValueError):
        read_facts(archive, required_tags(schema), ciks=[1])


@pytest.mark.parametrize('given', [320193, '320193', '0000320193',
                                   'CIK0000320193'])
def test_read_facts_accepts_any_cik_spelling(archive, schema, given):
    """Anything carrying the digits names the same filer."""
    facts = read_facts(archive, required_tags(schema), ciks=[given])
    assert set(facts['cik']) == {CIK}


def test_extract_drops_mostly_empty_rows(facts, schema):
    """A row where most datapoints are missing is a stray disclosure.

    Apple files plenty of them - a single balance sheet figure for a period
    the rest of the statement does not cover.
    """
    balance = next(s for s in schema if s.name == 'balance_sheet')
    wide = extract(facts, balance)

    assert not wide.empty
    assert wide[balance.datapoint_names].isna().mean(axis=1).le(0.5).all()


def test_collect_shape(collected):
    assert not collected.empty
    assert set(collected['cik']) == {CIK}

    for column in ('accn', 'fy', 'fp', 'form', 'start', 'end',
                   'period', 'filed', 'frame', 'is_latest'):
        assert column in collected.columns

    # periods are quarters or years, nothing in between
    assert collected['period'].dropna().isin([3.0, 12.0]).all()
    assert collected['is_latest'].any()


def test_collect_values(collected):
    latest = collected[collected['is_latest']]

    assert latest['revenue'].notna().any()
    assert latest['revenue'].dropna().gt(0).all()
    assert latest['net_income'].notna().any()
    assert latest['operating_cf'].notna().any()

    # balance sheet figures arrive as start/end pairs
    assert latest['end_total_assets'].dropna().gt(0).all()
    assert 'start_total_assets' in latest.columns


def test_collect_implies_quarterly_cash_flow(collected):
    """Cash flow is filed year-to-date, so Q2 and Q3 need the quarter implied.

    Without that repair those rows are six and nine months long, and the
    quarter-or-year filter drops them.
    """
    latest = collected[collected['is_latest']]
    for fp in ('Q1', 'Q2', 'Q3'):
        rows = latest[latest['fp'].eq(fp)]
        assert not rows.empty, fp
        assert rows['operating_cf'].notna().any(), fp


def test_collect_implies_a_fourth_quarter(collected):
    """No filing reports Q4 on its own - it is the year less three quarters."""
    assert 'Q4' in set(collected['fp'])

    q4 = collected[collected['fp'].eq('Q4')]
    assert q4['period'].eq(3.0).all()
    assert q4['revenue'].notna().any()
    # derived rows carry no SEC calendar label
    assert q4['frame'].isna().all()


def test_collect_keeps_every_filing_of_a_period(collected):
    """Restatements sit alongside the original rather than replacing it."""
    per_period = collected.groupby(
        ['cik', 'fp', 'start', 'end'], dropna=False)['filed'].nunique()
    assert per_period.max() > 1

    # exactly one row per period is flagged as the most recent
    flagged = collected.groupby(
        ['cik', 'start', 'end', 'period'], dropna=False)['is_latest'].sum()
    assert flagged.eq(1).all()
