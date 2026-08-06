"""The extraction schema, compiled into a shape the extractor can vectorize.

The JSON on disk describes each datapoint as an ordered list of extraction
methods, and two methods appear in it. `raw` names a single XBRL tag.
`sum_first_tag_found_per_sublist` names a list of tag groups and sums one value
out of each. Both share a structure once it is spelled out:

    datapoint   = first non-null of its alternatives
    alternative = sum of its terms, each term the first tag the filing reported

`raw` is the degenerate case - one term holding one tag - so compiling every
method into that shape lets the extractor evaluate any datapoint with the same
code, over whole columns rather than a row at a time.

Compiling also answers, once, which tags the schema can possibly reference.
That set is what the reader filters the archive down to, and it is far smaller
than what companies actually report: ~150 tags against a median of ~270
us-gaap tags per company.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Optional

# Datapoint columns are named for the datapoint. Calculations in the JSON refer
# to other datapoints through the column names the previous implementation gave
# them, which carried this prefix; it is stripped at compile time.
_CALC_TAG_PREFIX = 'fos_'

# Statements whose facts are measured at an instant rather than over a period.
# They are keyed by `end` alone, and their values become start/end pairs.
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
        terms: summed. Each term is a tuple of `Tag`s tried in order, the first
            one the filing reported winning. A single term of a single tag is
            the `raw` case.
        allow_null_components: whether a term that matched nothing may be
            treated as zero. False means the whole alternative fails instead,
            which is what keeps a sum like `total_liabilities` from being
            reported when only one of its halves was found.
    """

    terms: tuple
    allow_null_components: bool = True

    @property
    def tag_names(self) -> frozenset:
        return frozenset(tag.name for term in self.terms for tag in term)


@dataclass(frozen=True)
class Datapoint:
    """One output column: alternatives tried in order until one yields a value.

    Args:
        name: the output column name.
        alternatives: tried in order, first non-null wins.
    """

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
        overrides: per-CIK alternatives that are tried ahead of the shared ones.
            Keyed by CIK because a company keeps its CIK through the ticker
            changes and delistings that would silently break a ticker key.
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

        `filed` is in here deliberately: the same period is reported again in
        later filings, sometimes restated, and collapsing those would leave
        only the newest version - the one that was not knowable at the time.

        Point-in-time statements have no `start`; they are measured at `end`.
        """
        columns = ['cik', 'accn', 'fy', 'fp', 'form', 'start', 'end', 'filed']
        if self.is_pit:
            columns.remove('start')
        return columns

    @property
    def datapoint_names(self) -> list:
        """Output columns, mappings first, in the order the JSON declared.

        Calculations are almost always a fallback for a datapoint that is
        mapped too, so this is nearly the mappings alone - but a
        calculation-only datapoint is still an output column.
        """
        names = [dp.name for dp in self.mappings]
        return names + [dp.name for dp in self.calculations
                        if dp.name not in names]

    @property
    def required_tags(self) -> frozenset:
        """Every XBRL tag any mapping or override could read.

        Calculations are excluded: they read datapoint columns, which exist by
        the time they run, not tags off the filing.
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

    `is_calculation` only decides how tag names are read: a calculation names
    other datapoint columns rather than XBRL tags, under the prefix the
    previous implementation gave them.
    """
    return Datapoint(
        name=name,
        alternatives=tuple(
            _compile_alternative(m, is_calculation) for m in methods),
    )


def _compile_alternative(method: dict, is_calculation: bool) -> Alternative:
    """One method dict as a sum of terms.

    `raw` carries its tag inline and becomes a single term of a single tag;
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
        # Absent means permissive, matching how the previous implementation
        # read this flag.
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
