"""The extraction schema, compiled into a shape the extractor can vectorize.

The JSON describes each datapoint as an ordered list of extraction methods, of
which there are two: `raw` names one XBRL tag, and
`sum_first_tag_found_per_sublist` names tag groups and sums one value out of
each. Both compile to the same structure:

    datapoint   = first non-null of its alternatives
    alternative = sum of its terms, each term the first tag the filing reported

`raw` is the degenerate case - one term, one tag - so the extractor can
evaluate any datapoint with the same code, over whole columns.

Compiling also answers once which tags the schema can reference - the set the
reader filters the archive down to, ~150 against a median of ~270 us-gaap tags
per company.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Calculations name other datapoints by the column names the previous
# implementation gave them, which carried this prefix. Nothing else uses it.
_CALC_TAG_PREFIX = 'fos_'

# Statements measured at an instant rather than over a period. Keyed by `end`
# alone; their values become start/end pairs.
_PIT_STATEMENTS = frozenset({'balance_sheet', 'other'})

_SCHEMA_NAMES = ('base_mappings', 'base_calculations', 'override_mappings')


@dataclass(frozen=True)
class Tag:
    """One XBRL tag, and what to do with the value found under it."""

    name: str
    multiplier: float = 1.0
    ignore_if_zero: bool = False
    default: Optional[float] = None


@dataclass(frozen=True)
class Alternative:
    """Terms summed together to produce one candidate value.

    Args:
        terms: summed. Each is a tuple of `Tag`s tried in order, the first the
            filing reported winning. One term of one tag is the `raw` case.
        allow_null_components: whether a term that matched nothing counts as
            zero. False fails the whole alternative instead, which keeps a sum
            like `total_liabilities` off a row where only one half was found.
    """

    terms: tuple
    allow_null_components: bool = True


@dataclass(frozen=True)
class Datapoint:
    """One output column: alternatives tried in order, first non-null wins."""

    name: str
    alternatives: tuple


@dataclass(frozen=True)
class StatementSchema:
    """Everything needed to build one statement.

    Args:
        name: statement name, also the key used against the JSON files.
        mappings: datapoints read from XBRL tags.
        calculations: fallback datapoints derived from other columns, in
            dependency order - `total_liabilities` reads a column that
            `total_non_current_liabilities` writes, so order is load-bearing.
        overrides: per-CIK datapoints whose alternatives already carry the
            shared ones as a fallback. Keyed by CIK, which survives the ticker
            changes and delistings that would break a ticker key.
    """

    name: str
    mappings: tuple
    calculations: tuple
    overrides: dict = field(default_factory=dict)

    @property
    def is_pit(self) -> bool:
        """Whether facts here are measured at an instant, not over a period."""
        return self.name in _PIT_STATEMENTS

    @property
    def key_columns(self) -> list:
        """What identifies a row, before anything derived is added.

        `filed` is deliberate: periods are reported again in later filings,
        sometimes restated, and collapsing those would leave only the newest
        version - the one not knowable at the time. Point-in-time statements
        have no `start`; they are measured at `end`.
        """
        columns = ['cik', 'accn', 'fy', 'fp', 'form', 'start', 'end', 'filed']
        if self.is_pit:
            columns.remove('start')
        return columns

    @property
    def datapoint_names(self) -> list:
        """Output columns, mappings first, in the order the JSON declared."""
        return list(dict.fromkeys(
            dp.name for dp in self.mappings + self.calculations))

    @property
    def required_tags(self) -> frozenset:
        """Every XBRL tag any mapping or override could read.

        Calculations are excluded - they read datapoint columns, not tags.
        """
        datapoints = self.mappings + tuple(
            dp for dps in self.overrides.values() for dp in dps)
        return frozenset(
            tag.name for dp in datapoints for alternative in dp.alternatives
            for term in alternative.terms for tag in term)


def load_schema(schema_dir: str) -> tuple:
    """Compile the JSON schema files in `schema_dir`, one entry per statement."""
    mappings, calculations, overrides = (
        json.loads((Path(schema_dir) / f'{name}.json').read_text())
        for name in _SCHEMA_NAMES)

    return tuple(
        _compile_statement(name, datapoints,
                           calculations.get(name, {}), overrides.get(name, {}))
        for name, datapoints in mappings.items())


def required_tags(statements: tuple) -> frozenset:
    """Every tag any statement could read, which is what the reader keeps."""
    return frozenset().union(*(s.required_tags for s in statements))


def _compile_statement(
    name: str,
    mappings: dict,
    calculations: dict,
    overrides: dict,
) -> StatementSchema:
    compiled = tuple(
        _compile_datapoint(dp_name, methods)
        for dp_name, methods in mappings.items())
    base = {dp.name: dp.alternatives for dp in compiled}

    return StatementSchema(
        name=name,
        mappings=compiled,
        calculations=tuple(
            _compile_datapoint(dp_name, methods)
            for dp_name, methods in calculations.items()),
        overrides={
            # The shared alternatives are appended here rather than consulted
            # at extraction time, so an override still falls back to them.
            cik: tuple(
                _compile_datapoint(dp_name, methods,
                                   fallback=base.get(dp_name, ()))
                for dp_name, methods in datapoints.items())
            for cik, datapoints in overrides.items()
        },
    )


def _compile_datapoint(name: str, methods: list, fallback: tuple = ()) -> Datapoint:
    return Datapoint(
        name=name,
        alternatives=tuple(_compile_alternative(m) for m in methods) + fallback,
    )


def _compile_alternative(method: dict) -> Alternative:
    """One method dict as a sum of terms.

    `raw` carries its tag inline and becomes one term of one tag;
    `sum_first_tag_found_per_sublist` carries `tag_li`, a list of terms whose
    inner dicts are themselves `raw`.
    """
    if 'tag_li' in method:
        terms = tuple(tuple(_compile_tag(t) for t in term)
                      for term in method['tag_li'])
    else:
        terms = ((_compile_tag(method),),)

    return Alternative(
        terms=terms,
        # Absent means permissive, as the previous implementation read it.
        allow_null_components=method.get('allow_null_components', True),
    )


def _compile_tag(method: dict) -> Tag:
    return Tag(
        name=method['tag'].removeprefix(_CALC_TAG_PREFIX),
        multiplier=float(method.get('multiplier', 1.0)),
        ignore_if_zero=bool(method.get('ignore_if_zero', False)),
        default=method.get('default'),
    )
