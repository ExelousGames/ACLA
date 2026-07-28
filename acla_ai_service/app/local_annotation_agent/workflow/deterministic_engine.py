"""Generic interpreter for declarative deterministic requirements."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple


InputKind = Literal["iloc", "range"]
RangeTuple = Tuple[int, int]
MISSING = object()
_TAG_OBJECT_KEY_ORDERS = (
    ("player", "expert"),
    ("start", "end"),
)


@dataclass(frozen=True, order=True)
class InclusiveRange:
    start: int
    end: int

    def __post_init__(self) -> None:
        if int(self.start) > int(self.end):
            raise ValueError("range start must not exceed range end")
        object.__setattr__(self, "start", int(self.start))
        object.__setattr__(self, "end", int(self.end))

    @classmethod
    def envelope(cls, ranges: Iterable["InclusiveRange"]) -> Optional["InclusiveRange"]:
        values = list(ranges)
        if not values:
            return None
        return cls(min(value.start for value in values), max(value.end for value in values))

    def as_tuple(self) -> RangeTuple:
        return self.start, self.end

    def contains(self, other: "InclusiveRange") -> bool:
        return self.start <= other.start and other.end <= self.end


@dataclass(frozen=True)
class ResolvedInput:
    tag: str
    kind: InputKind
    value: int | InclusiveRange
    evidence_range: InclusiveRange


InputResolver = Callable[[Any, InclusiveRange], Optional[ResolvedInput]]
FactCalculator = Callable[[Any, Sequence[ResolvedInput]], Any]


@dataclass(frozen=True)
class InputDefinition:
    kind: InputKind
    resolve: InputResolver


@dataclass(frozen=True)
class FactDefinition:
    input_kinds: Tuple[InputKind, ...]
    calculate: FactCalculator


class InputRegistry:
    def __init__(
        self,
        definitions: Mapping[str, InputDefinition] | Iterable[Tuple[str, InputDefinition]],
    ) -> None:
        self._definitions: Dict[str, InputDefinition] = {}
        entries = definitions.items() if isinstance(definitions, Mapping) else definitions
        for name, definition in entries:
            if name in self._definitions:
                raise ValueError(f"duplicate input strategy {name!r}")
            self._definitions[name] = definition

    def names(self) -> frozenset[str]:
        return frozenset(self._definitions)

    def get(self, name: str) -> Optional[InputDefinition]:
        return self._definitions.get(name)


class FactRegistry:
    def __init__(
        self,
        definitions: Mapping[str, FactDefinition] | Iterable[Tuple[str, FactDefinition]],
    ) -> None:
        self._definitions: Dict[str, FactDefinition] = {}
        entries = definitions.items() if isinstance(definitions, Mapping) else definitions
        for name, definition in entries:
            if name in self._definitions:
                raise ValueError(f"duplicate fact strategy {name!r}")
            self._definitions[name] = definition

    def names(self) -> frozenset[str]:
        return frozenset(self._definitions)

    def get(self, name: str) -> Optional[FactDefinition]:
        return self._definitions.get(name)


@dataclass(frozen=True)
class PredicateSpec:
    tags: Tuple[str, ...]
    combine_tags: bool
    fact: str
    operator: str
    expected: Any
    evidence_only: bool = False


@dataclass(frozen=True)
class BranchSpec:
    predicates: Tuple[PredicateSpec, ...]


@dataclass(frozen=True)
class RequirementSpec:
    enabled: bool
    branches: Tuple[BranchSpec, ...]


@dataclass
class PredicateEvaluation:
    passed: bool
    text: str
    fact: str
    evidence_range: Optional[InclusiveRange] = None


@dataclass
class RequirementBranchEvaluation:
    branch: int
    predicates: List[PredicateEvaluation] = field(default_factory=list)

    @property
    def passed(self) -> List[str]:
        return [value.text for value in self.predicates if value.passed]

    @property
    def failed(self) -> List[str]:
        return [value.text for value in self.predicates if not value.passed]

    @property
    def evidence_range(self) -> Optional[InclusiveRange]:
        return InclusiveRange.envelope(
            value.evidence_range
            for value in self.predicates
            if value.passed and value.evidence_range is not None
        )


@dataclass
class RequirementEvaluation:
    matched: bool
    branch: Optional[int] = None
    passed: List[str] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)
    matched_branches: List[RequirementBranchEvaluation] = field(default_factory=list)


@dataclass
class LabelEvaluation:
    labels: List[str]
    evaluations: Dict[str, RequirementEvaluation]
    conflicts: List[Tuple[str, str]] = field(default_factory=list)


SUPPORTED_OPERATORS = frozenset({
    "eq", "neq", "in", "not_in", "lt", "lte", "gt", "gte",
    "between", "contains", "exists",
})


def compare(actual: Any, operator: str, expected: Any = None) -> bool:
    if operator not in SUPPORTED_OPERATORS:
        return False
    if operator == "exists":
        return actual is not MISSING and (bool(actual is not None) == bool(expected))
    if actual is MISSING or actual is None:
        return False
    try:
        comparisons = {
            "eq": lambda: actual == expected,
            "neq": lambda: actual != expected,
            "in": lambda: actual in expected,
            "not_in": lambda: actual not in expected,
            "lt": lambda: actual < expected,
            "lte": lambda: actual <= expected,
            "gt": lambda: actual > expected,
            "gte": lambda: actual >= expected,
            "between": lambda: expected[0] <= actual <= expected[1],
            "contains": lambda: expected in actual,
        }
        return bool(comparisons[operator]())
    except (TypeError, ValueError, IndexError, KeyError):
        return False


def parse_requirements(requirements: Mapping[str, Any]) -> RequirementSpec:
    branches: List[BranchSpec] = []
    for branch in requirements.get("any_of") or []:
        predicates: List[PredicateSpec] = []
        if not isinstance(branch, Mapping):
            continue
        for raw in branch.get("all_of") or []:
            if not isinstance(raw, Mapping):
                continue
            inputs = raw.get("inputs")
            condition = raw.get("condition")
            if not isinstance(inputs, Mapping) or not isinstance(condition, Mapping):
                predicates.append(PredicateSpec((), False, "", "", None))
                continue
            raw_tags = inputs.get("tags")
            if isinstance(raw_tags, list):
                tags = tuple(str(tag) for tag in raw_tags)
                combine_tags = True
            elif isinstance(raw_tags, Mapping):
                key_order = next(
                    (
                        keys for keys in _TAG_OBJECT_KEY_ORDERS
                        if set(raw_tags) == set(keys)
                    ),
                    (),
                )
                tags = tuple(
                    str(raw_tags.get(role) or "") for role in key_order
                )
                combine_tags = False
            else:
                tags = ()
                combine_tags = False
            predicates.append(PredicateSpec(
                tags=tags,
                combine_tags=combine_tags,
                fact=str(condition.get("fact") or ""),
                operator=str(condition.get("operator") or ""),
                expected=condition.get("value"),
                evidence_only=not condition,
            ))
        branches.append(BranchSpec(tuple(predicates)))
    return RequirementSpec(requirements.get("enabled") is not False, tuple(branches))


class RequirementInterpreter:
    def __init__(self, inputs: InputRegistry, facts: FactRegistry) -> None:
        self.inputs = inputs
        self.facts = facts

    def evaluate(
        self, requirements: Mapping[str, Any], context: Any, scope: InclusiveRange,
    ) -> RequirementEvaluation:
        spec = parse_requirements(requirements)
        if not spec.enabled:
            return RequirementEvaluation(False, failed=["label disabled"])
        if not spec.branches:
            return RequirementEvaluation(False, failed=["no valid requirement branches"])

        closest: Optional[RequirementBranchEvaluation] = None
        matches: List[RequirementBranchEvaluation] = []
        for index, branch in enumerate(spec.branches):
            if not branch.predicates:
                continue
            candidate = RequirementBranchEvaluation(index)
            for predicate in branch.predicates:
                candidate.predicates.append(self._evaluate_predicate(predicate, context, scope))
            if not candidate.failed:
                matches.append(candidate)
            elif closest is None or (len(candidate.failed), -len(candidate.passed)) < (
                len(closest.failed), -len(closest.passed)
            ):
                closest = candidate

        if matches:
            first = matches[0]
            return RequirementEvaluation(
                True, first.branch, first.passed, [], matches,
            )
        if closest is not None:
            return RequirementEvaluation(
                False, closest.branch, closest.passed, closest.failed,
            )
        return RequirementEvaluation(False, failed=["facts unavailable"])

    def _evaluate_predicate(
        self, predicate: PredicateSpec, context: Any, scope: InclusiveRange,
    ) -> PredicateEvaluation:
        definition = None if predicate.evidence_only else self.facts.get(predicate.fact)
        if not predicate.evidence_only and definition is None:
            return PredicateEvaluation(False, f"{predicate.fact}: unknown fact", predicate.fact)
        resolved: List[ResolvedInput] = []
        for tag in predicate.tags:
            value = context.resolve_input(tag, scope, self.inputs)
            if value is None:
                name = "evidence" if predicate.evidence_only else predicate.fact
                return PredicateEvaluation(
                    False, f"{name}: unavailable (missing input {tag})", predicate.fact,
                )
            resolved.append(value)
        evidence_range = InclusiveRange.envelope(
            value.evidence_range for value in resolved
        )
        if predicate.evidence_only:
            return PredicateEvaluation(
                True, "evidence: inputs resolved", "", evidence_range,
            )
        condition_inputs = resolved
        if predicate.combine_tags and len(resolved) > 1:
            assert evidence_range is not None
            condition_inputs = [ResolvedInput(
                "combined_range", "range", evidence_range, evidence_range,
            )]
        kinds = tuple(value.kind for value in condition_inputs)
        assert definition is not None
        if kinds != definition.input_kinds:
            return PredicateEvaluation(
                False,
                f"{predicate.fact}: unavailable (expected {definition.input_kinds}, got {kinds})",
                predicate.fact,
            )
        actual = context.calculate_fact(
            predicate.fact, definition, condition_inputs,
        )
        value_text = "unavailable" if actual is MISSING else repr(actual)
        passed = compare(actual, predicate.operator, predicate.expected)
        return PredicateEvaluation(
            passed,
            f"{predicate.fact}: {value_text}",
            predicate.fact,
            evidence_range,
        )


def validate_requirements(
    requirements: Mapping[str, Any], inputs: InputRegistry, facts: FactRegistry,
) -> List[str]:
    errors: List[str] = []
    enabled = requirements.get("enabled") is not False
    branches = requirements.get("any_of")
    if enabled and (not isinstance(branches, list) or not branches):
        return ["active label has no any_of branches"]
    for branch_index, branch in enumerate(branches or []):
        predicates = branch.get("all_of") if isinstance(branch, Mapping) else None
        if enabled and (not isinstance(predicates, list) or not predicates):
            errors.append(f"branch {branch_index} has no all_of predicates")
            continue
        for predicate_index, predicate in enumerate(predicates or []):
            prefix = f"branch {branch_index} predicate {predicate_index}"
            if not isinstance(predicate, Mapping) or set(predicate) != {"inputs", "condition"}:
                errors.append(f"{prefix}: expected inputs and condition")
                continue
            raw_inputs = predicate.get("inputs")
            condition = predicate.get("condition")
            if not isinstance(raw_inputs, Mapping) or set(raw_inputs) != {"tags"}:
                errors.append(f"{prefix}: inputs must contain only tags")
                continue
            if not isinstance(condition, Mapping) or set(condition) not in (
                set(), {"fact", "operator", "value"},
            ):
                errors.append(
                    f"{prefix}: condition must be empty or contain fact, operator, and value"
                )
                continue
            raw_tags = raw_inputs.get("tags")
            combine_tags = isinstance(raw_tags, list)
            if combine_tags:
                tags = raw_tags
                valid_tags = (
                    bool(tags)
                    and all(isinstance(tag, str) and tag for tag in tags)
                )
            elif isinstance(raw_tags, Mapping):
                key_order = next(
                    (
                        keys for keys in _TAG_OBJECT_KEY_ORDERS
                        if set(raw_tags) == set(keys)
                    ),
                    (),
                )
                valid_tags = (
                    bool(key_order)
                    and all(
                        isinstance(raw_tags.get(role), str) and raw_tags.get(role)
                        for role in key_order
                    )
                )
                tags = [
                    raw_tags.get(role) for role in key_order
                ]
            else:
                tags = []
                valid_tags = False
            if not valid_tags:
                errors.append(
                    f"{prefix}: tags must be a non-empty string list "
                    "or a player/expert or start/end string object"
                )
                continue
            kinds: List[InputKind] = []
            for tag in tags:
                assert isinstance(tag, str)
                input_definition = inputs.get(tag)
                if input_definition is None:
                    errors.append(f"{prefix}: unknown input tag {tag!r}")
                else:
                    kinds.append(input_definition.kind)
            if not condition:
                continue
            fact_name = condition.get("fact")
            fact = facts.get(fact_name) if isinstance(fact_name, str) else None
            if fact is None:
                errors.append(f"{prefix}: unknown fact {fact_name!r}")
            condition_kinds = (
                ("range",) if combine_tags and len(tags) > 1 else tuple(kinds)
            )
            if fact is not None and condition_kinds != fact.input_kinds:
                errors.append(
                    f"{prefix}: {fact_name!r} expects {fact.input_kinds}, "
                    f"got {condition_kinds}"
                )
            operator = condition.get("operator")
            if operator not in SUPPORTED_OPERATORS:
                errors.append(f"{prefix}: unknown operator {operator!r}")
    return errors


__all__ = [
    "BranchSpec", "FactDefinition", "FactRegistry", "InclusiveRange",
    "InputDefinition", "InputRegistry", "LabelEvaluation", "MISSING",
    "PredicateEvaluation", "PredicateSpec", "RequirementBranchEvaluation",
    "RequirementEvaluation", "RequirementInterpreter", "RequirementSpec",
    "ResolvedInput", "SUPPORTED_OPERATORS", "compare", "parse_requirements",
    "validate_requirements",
]
