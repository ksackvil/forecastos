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
import os
from dataclasses import dataclass, field
from typing import Optional

# Calculations name other datapoints by the column names the previous
# implementation gave them, which carried this prefix. Stripped at compile time.
_CALC_TAG_PREFIX = 'fos_'

# Statements measured at an instant rather than over a period. Keyed by `end`
# alone; their values become start/end pairs.
_PIT_STATEMENTS = frozenset({'balance_sheet', 'other'})

_SCHEMA_FILES = {
    'base_mappings': 'base_mappings.json',
    'base_calculations': 'base_calculations.json',
    'override_mappings': 'override_mappings.json',
}


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

    @property
    def tag_names(self) -> frozenset:
        return frozenset(tag.name for term in self.terms for tag in term)


@dataclass(frozen=True)
class Datapoint:
    """One output column: alternatives tried in order, first non-null wins."""

    name: str
    alternatives: tuple

    @property
    def tag_names(self) -> frozenset:
        return frozenset().union(*(a.tag_names for a in self.alternatives))


@dataclass(frozen=True)
class StatementSchema:
    """Everything needed to build one statement.

    Args:
        name: statement name, also the key used against the JSON files.
        mappings: datapoints read from XBRL tags.
        calculations: fallback datapoints derived from other columns, in
            dependency order - `total_liabilities` reads a column that
            `total_non_current_liabilities` writes, so order is load-bearing.
        overrides: per-CIK alternatives tried ahead of the shared ones. Keyed
            by CIK, which survives the ticker changes and delistings that would
            break a ticker key.
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
        """Output columns, mappings first, in the order the JSON declared.

        Calculations are nearly always a fallback for an already-mapped
        datapoint, but a calculation-only one is still an output column.
        """
        names = [dp.name for dp in self.mappings]
        return names + [dp.name for dp in self.calculations
                        if dp.name not in names]

    @property
    def required_tags(self) -> frozenset:
        """Every XBRL tag any mapping or override could read.

        Calculations are excluded - they read datapoint columns, not tags.
        """
        tags = frozenset().union(
            *(dp.tag_names for dp in self.mappings)) if self.mappings else frozenset()
        for datapoints in self.overrides.values():
            for dp in datapoints:
                tags |= dp.tag_names
        return tags

    def alternatives_for(self, name: str, cik: str) -> tuple:
        """Alternatives for one datapoint, with `cik`'s overrides tried first."""
        base = next(
            (dp.alternatives for dp in self.mappings if dp.name == name), ())
        override = next(
            (dp.alternatives for dp in self.overrides.get(cik, ())
             if dp.name == name), ())
        return override + base


@dataclass(frozen=True)
class Schema:
    """The compiled schema for every statement."""

    statements: tuple

    @classmethod
    def from_dir(cls, schema_dir: str) -> 'Schema':
        """Compile the JSON schema files in `schema_dir`."""
        raw = {}
        for key, filename in _SCHEMA_FILES.items():
            with open(os.path.join(schema_dir, filename)) as f:
                raw[key] = json.load(f)

        names = list(raw['base_mappings'])
        return cls(statements=tuple(
            _compile_statement(
                name,
                raw['base_mappings'].get(name, {}),
                raw['base_calculations'].get(name, {}),
                raw['override_mappings'].get(name, {}),
            )
            for name in names
        ))

    @property
    def required_tags(self) -> frozenset:
        """Every tag any statement could read, which is what the reader keeps."""
        return frozenset().union(*(s.required_tags for s in self.statements))

    def __iter__(self):
        return iter(self.statements)


def _compile_statement(
    name: str,
    mappings: dict,
    calculations: dict,
    overrides: dict,
) -> StatementSchema:
    return StatementSchema(
        name=name,
        mappings=tuple(
            _compile_datapoint(dp_name, methods)
            for dp_name, methods in mappings.items()
        ),
        calculations=tuple(
            _compile_datapoint(dp_name, methods, is_calculation=True)
            for dp_name, methods in calculations.items()
        ),
        overrides={
            cik: tuple(
                _compile_datapoint(dp_name, methods)
                for dp_name, methods in datapoints.items()
            )
            for cik, datapoints in overrides.items()
        },
    )


def _compile_datapoint(
    name: str,
    methods: list,
    is_calculation: bool = False,
) -> Datapoint:
    """Compile one datapoint.

    `is_calculation` only affects how tag names are read - a calculation names
    datapoint columns, under the prefix the previous implementation used.
    """
    return Datapoint(
        name=name,
        alternatives=tuple(
            _compile_alternative(m, is_calculation) for m in methods),
    )


def _compile_alternative(method: dict, is_calculation: bool) -> Alternative:
    """One method dict as a sum of terms.

    `raw` carries its tag inline and becomes one term of one tag;
    `sum_first_tag_found_per_sublist` carries `tag_li`, a list of terms whose
    inner dicts are themselves `raw`.
    """
    if 'tag_li' in method:
        terms = tuple(
            tuple(_compile_tag(t, is_calculation) for t in term)
            for term in method['tag_li']
        )
    else:
        terms = ((_compile_tag(method, is_calculation),),)

    return Alternative(
        terms=terms,
        # Absent means permissive, as the previous implementation read it.
        allow_null_components=method.get('allow_null_components', True),
    )


def _compile_tag(method: dict, is_calculation: bool) -> Tag:
    tag = method['tag']
    if is_calculation and tag.startswith(_CALC_TAG_PREFIX):
        tag = tag[len(_CALC_TAG_PREFIX):]

    return Tag(
        name=tag,
        multiplier=float(method.get('multiplier', 1.0)),
        ignore_if_zero=bool(method.get('ignore_if_zero', False)),
        default=method.get('default'),
    )
