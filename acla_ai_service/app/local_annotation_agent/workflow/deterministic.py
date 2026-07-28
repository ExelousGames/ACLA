"""Deterministic telemetry-to-label workflow orchestration.

The catalogs describe selection policy. Input and fact strategies resolve only
the data declared by each predicate, and the generic interpreter carries that
input provenance into annotation ranges.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from app.internal_knowledge_base import skills
from app.internal_knowledge_base.label_lookup import get_label
from app.local_annotation_agent.workflow.deterministic_engine import (
    HalfOpenRange,
    LabelEvaluation,
    RequirementBranchEvaluation,
    RequirementEvaluation,
    RequirementInterpreter,
    SUPPORTED_OPERATORS,
    validate_requirements,
)
from app.local_annotation_agent.workflow.deterministic_facts import (
    EvaluationContext,
    FACT_REGISTRY,
    INPUT_REGISTRY,
    smooth_telemetry,
)
from app.local_annotation_agent.workflow.results import AnnotationResult, LapAnnotationResult
from app.shared.labels import BEHAVIOR_LABELS, LABEL_CATEGORIES, LABEL_MAPPING


INTERPRETER = RequirementInterpreter(INPUT_REGISTRY, FACT_REGISTRY)


def _requirements_for(
    label_id: str, doc: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    if doc is not None:
        requirements = doc.get("selection_requirements")
        if isinstance(requirements, dict):
            return dict(requirements)
        reference = doc.get("selection_requirements_ref")
        if isinstance(reference, str):
            value = skills.get(reference, {})
            if isinstance(value, dict):
                return dict(value)
    for path in (
        "lap_annotation.selection_requirements",
        "sub_label_annotation.sub_label_selection_requirements",
        "sub_label_annotation.segment_type_selection_requirements",
    ):
        requirements = skills.get(path, {})
        if isinstance(requirements, dict) and isinstance(
            requirements.get(label_id), dict,
        ):
            return dict(requirements[label_id])
    return {}


def evaluate_requirements(
    requirements: Mapping[str, Any], context: EvaluationContext,
    scope: HalfOpenRange,
) -> RequirementEvaluation:
    return INTERPRETER.evaluate(requirements, context, scope)


def evaluate_labels(
    label_ids: Iterable[str], context: EvaluationContext, scope: HalfOpenRange,
) -> LabelEvaluation:
    evaluations: Dict[str, RequirementEvaluation] = {}
    matched: List[str] = []
    docs: Dict[str, Dict[str, Any]] = {}
    for label_id in label_ids:
        doc = get_label(label_id)
        docs[label_id] = doc or {"id": label_id}
        evaluation = evaluate_requirements(_requirements_for(label_id, doc), context, scope)
        evaluations[label_id] = evaluation
        if evaluation.matched:
            matched.append(label_id)

    conflicts: List[Tuple[str, str]] = []
    suppressed: set[str] = set()
    matched_set = set(matched)
    for label_id in matched:
        for other in docs[label_id].get("exclusive_with") or []:
            if other not in matched_set:
                continue
            pair = tuple(sorted((label_id, other)))
            if pair not in conflicts:
                conflicts.append(pair)
            suppressed.update(pair)
    return LabelEvaluation(
        [label for label in matched if label not in suppressed], evaluations, conflicts,
    )


def validate_catalog() -> List[str]:
    """Return structural and registry errors in deterministic requirements."""
    errors: List[str] = []
    main_labels = {doc["id"]: doc for doc in skills.iter("lap_annotation.labels")}
    sub_requirements = skills.get(
        "sub_label_annotation.sub_label_selection_requirements", {},
    )
    segment_type_requirements = skills.get(
        "sub_label_annotation.segment_type_selection_requirements", {},
    )
    sub_requirements = sub_requirements if isinstance(sub_requirements, dict) else {}
    segment_type_requirements = (
        segment_type_requirements
        if isinstance(segment_type_requirements, dict)
        else {}
    )
    parent_by_child = {
        child: parent
        for parent in BEHAVIOR_LABELS
        for child in LABEL_CATEGORIES.get(parent, [])
    }
    sub_labels = {
        label_id: {
            "id": label_id,
            "type": "sub",
            "parent": parent_by_child.get(label_id),
        }
        for label_id in sub_requirements
    }
    segment_type_labels = {
        label_id: {"id": label_id, "type": "segment_type", "parent": None}
        for label_id in segment_type_requirements
    }
    non_main_labels = {**sub_labels, **segment_type_labels}
    duplicate_ids = set(main_labels) & set(non_main_labels)
    if duplicate_ids:
        errors.append(f"label IDs exist in both catalogs: {sorted(duplicate_ids)}")
    labels = {**main_labels, **non_main_labels}
    requirement_groups = (
        ("lap", skills.get("lap_annotation.selection_requirements", {}), main_labels),
        ("sub-label", sub_requirements, sub_labels),
        ("segment-type", segment_type_requirements, segment_type_labels),
    )
    if any(doc.get("type") != "main" for doc in main_labels.values()):
        errors.append("lap label catalog contains non-main labels")
    unknown_sub_labels = sorted(set(sub_labels) - set(parent_by_child))
    if unknown_sub_labels:
        errors.append(
            f"sub-label requirement IDs have no shared parent: {unknown_sub_labels}"
        )
    unknown_labels = sorted(set(non_main_labels) - set(LABEL_MAPPING))
    if unknown_labels:
        errors.append(f"requirement IDs have no shared label name: {unknown_labels}")
    for group_name, requirements, expected in requirement_groups:
        if not isinstance(requirements, dict) or set(requirements) != set(expected):
            errors.append(f"{group_name} requirement IDs do not exactly match label IDs")

    for label_id, doc in labels.items():
        requirements = _requirements_for(label_id, get_label(label_id) or doc)
        errors.extend(
            f"{label_id}: {error}"
            for error in validate_requirements(requirements, INPUT_REGISTRY, FACT_REGISTRY)
        )
        parent = doc.get("parent")
        if parent is not None and parent not in labels:
            errors.append(f"{label_id}: unknown parent {parent!r}")
        for other in doc.get("exclusive_with") or []:
            if other not in labels:
                errors.append(f"{label_id}: unknown exclusive label {other!r}")
    return errors


def _reason_from_passed(
    label_id: str, passed: Iterable[str], range_: HalfOpenRange,
) -> str:
    return "; ".join([
        f"{label_id} selected for iloc range [{range_.start}, {range_.end})",
        *(f"Passed — {fact}" for fact in passed),
    ])


def _detailed_reason_from_passed(
    label_id: str, passed: Iterable[str], range_: HalfOpenRange,
) -> str:
    return "\n".join([
        f"{label_id} selected for iloc range [{range_.start}, {range_.end})",
        "Evidence:",
        *(f"- {fact}" for fact in passed),
    ])


def _passed_with_evidence(evaluation: RequirementBranchEvaluation) -> List[str]:
    annotated: List[str] = []
    for predicate in evaluation.predicates:
        if not predicate.passed:
            continue
        range_ = predicate.evidence_range
        location = (
            f"range [{range_.start}, {range_.end})" if range_ is not None
            else "range unavailable"
        )
        annotated.append(f"{predicate.text} — {location}")
    return annotated


def _resolve_circuit_sections(
    df: pd.DataFrame, circuit_id: str, section_id: str, start: int, end: int,
    opponent_interaction: Optional[dict],
) -> Tuple[str, List[str]]:
    primary_id = (
        section_id
        if section_id in LABEL_MAPPING and section_id != "interaction_window"
        else None
    )
    context_ids = []
    if isinstance(opponent_interaction, dict):
        for context in opponent_interaction.get("section_context") or []:
            candidate = (
                context.get("circuit_section_id") if isinstance(context, dict) else None
            )
            if candidate in LABEL_MAPPING and candidate not in context_ids:
                context_ids.append(candidate)
    try:
        from app.shared.annotation_agent_tools import locate_circuit_section

        content = locate_circuit_section(
            df, circuit_id, start, end,
        )
    except Exception:
        content = {}
    best = content.get("best_match") or {}
    candidate = best.get("label_id") if isinstance(best, dict) else None
    overlap_ids = [
        str(match["label_id"])
        for match in content.get("top_matches") or []
        if isinstance(match, dict) and match.get("label_id") in LABEL_MAPPING
    ]
    candidate_ids = [*([primary_id] if primary_id else []), *context_ids, *overlap_ids]
    candidate_ids = [
        value for index, value in enumerate(candidate_ids)
        if value not in candidate_ids[:index]
    ]
    if primary_id:
        return primary_id, candidate_ids
    if len(context_ids) == 1:
        return context_ids[0], candidate_ids
    if candidate in LABEL_MAPPING:
        return str(candidate), candidate_ids or [str(candidate)]
    if candidate_ids:
        return candidate_ids[0], candidate_ids
    return section_id, []


def _first_branch_range(
    evaluation: RequirementEvaluation, fallback: HalfOpenRange,
) -> HalfOpenRange:
    if evaluation.matched_branches:
        return evaluation.matched_branches[0].evidence_range or fallback
    return fallback


def _is_far_from_expert_in_pit(
    context: EvaluationContext, scope: HalfOpenRange,
) -> bool:
    overlap_names = [
        LABEL_MAPPING[value]
        for value in context.overlap_section_ids
        if value in LABEL_MAPPING
    ]
    if "Pit" not in overlap_names:
        return False
    requirements = {
        "enabled": True,
        "any_of": [{"all_of": [{
            "inputs": {"tags": ["trajectory_comparison_range"]},
            "condition": {
                "fact": "find_trajectory_peak_offset",
                "operator": "gte",
                "value": 10.0,
            },
        }]}],
    }
    return evaluate_requirements(requirements, context, scope).matched


def calculate_lap_annotation(
    df: pd.DataFrame, *, lap_start: int, lap_end: int, section_id: str,
    section_start: int, section_end: int, circuit_id: str,
    section_split_basis: Optional[str] = None,
    opponent_interaction: Optional[dict] = None,
) -> LapAnnotationResult:
    del lap_start, lap_end
    session = (
        "racing"
        if opponent_interaction or "interaction" in str(section_split_basis or "")
        else "practice"
    )
    eligible = skills.get(
        f"lap_annotation.behavior_parent_label_ids.eligible_by_session.{session}", [],
    )
    eligible = [
        label for label in eligible if label in skills.get("lap_annotation.labels", {})
    ]
    telemetry = smooth_telemetry(df)
    resolved_section_id, overlap_section_ids = _resolve_circuit_sections(
        telemetry, circuit_id, section_id, section_start, section_end,
        opponent_interaction,
    )
    scope = HalfOpenRange(section_start, section_end)
    context = EvaluationContext(
        telemetry, section_id=resolved_section_id,
        overlap_section_ids=tuple(overlap_section_ids),
    )
    if _is_far_from_expert_in_pit(context, scope):
        eligible = [label for label in eligible if label != "RM"]
    evaluated = evaluate_labels(eligible, context, scope)
    behavior = evaluated.labels
    if "PS" in behavior:
        resolved_section_id = next(
            (
                candidate for candidate in overlap_section_ids
                if LABEL_MAPPING.get(candidate) == "Pit"
            ),
            resolved_section_id,
        )
    sub_requirements = skills.get(
        "sub_label_annotation.sub_label_selection_requirements", {},
    )
    child_ids = [
        label_id
        for parent in behavior
        for label_id in LABEL_CATEGORIES.get(parent, [])
        if label_id in sub_requirements
    ]
    children = evaluate_labels(child_ids, context, scope)
    resolved_children: List[Tuple[str, RequirementBranchEvaluation, HalfOpenRange]] = []
    for label in children.labels:
        for branch in children.evaluations[label].matched_branches:
            range_ = branch.evidence_range
            if range_ == scope:
                resolved_children.append((label, branch, range_))
                break

    label_ids = (
        [
            circuit_id, resolved_section_id, *behavior,
            *(label for label, _, _ in resolved_children),
        ]
        if behavior else []
    )
    notes = [
        _reason_from_passed(
            label,
            evaluated.evaluations[label].passed,
            _first_branch_range(evaluated.evaluations[label], scope),
        )
        for label in behavior
    ]
    notes.extend(
        _reason_from_passed(label, branch.passed, range_)
        for label, branch, range_ in resolved_children
    )
    rejected = [
        {
            "value": label_id,
            "reason": "; ".join([
                *(f"Passed — {fact}" for fact in evaluation.passed),
                *(f"Failed — {fact}" for fact in evaluation.failed),
            ]),
        }
        for label_id, evaluation in evaluated.evaluations.items()
        if not evaluation.matched
        or any(label_id in pair for pair in evaluated.conflicts)
    ]
    rejected.extend(
        {
            "value": " / ".join(pair),
            "label_ids": list(pair),
            "reason": "exclusive deterministic matches",
        }
        for pair in children.conflicts
    )
    return LapAnnotationResult(
        section_id=resolved_section_id,
        start_index=scope.start,
        end_index=scope.end,
        label_ids=[
            label for index, label in enumerate(label_ids)
            if label and label not in label_ids[:index]
        ],
        reasoning="\n".join(notes)
        or "No behavior label satisfied a complete requirement branch.",
        submitted=True,
        rejected_proposals=rejected,
        transcript="deterministic label evaluation",
        tool_calls=0,
    )


def _merge_label_annotations(annotations: Sequence[dict]) -> List[dict]:
    by_label: Dict[str, List[dict]] = {}
    for annotation in annotations:
        by_label.setdefault(str(annotation["label_id"]), []).append(annotation)

    merged: List[dict] = []
    for label_id, proposals in by_label.items():
        ordered = sorted(
            proposals,
            key=lambda item: (int(item["start_index"]), int(item["end_index"])),
        )
        current: Optional[dict] = None
        for proposal in ordered:
            if current is None or int(proposal["start_index"]) >= int(current["end_index"]):
                if current is not None:
                    merged.append(current)
                current = {
                    "label_id": label_id,
                    "start_index": int(proposal["start_index"]),
                    "end_index": int(proposal["end_index"]),
                    "passed": list(dict.fromkeys(proposal["passed"])),
                }
                continue
            current["end_index"] = max(
                int(current["end_index"]), int(proposal["end_index"]),
            )
            current["passed"] = list(dict.fromkeys([
                *current["passed"], *proposal["passed"],
            ]))
        if current is not None:
            merged.append(current)

    result = []
    for proposal in sorted(
        merged,
        key=lambda item: (
            int(item["start_index"]), int(item["end_index"]), item["label_id"],
        ),
    ):
        range_ = HalfOpenRange(proposal["start_index"], proposal["end_index"])
        result.append({
            "label_id": proposal["label_id"],
            "start_index": range_.start,
            "end_index": range_.end,
            "reasoning": _detailed_reason_from_passed(
                proposal["label_id"], proposal["passed"], range_,
            ),
        })
    return result


def calculate_detailed_annotation(
    df: pd.DataFrame, *, parent_start: int, parent_end: int,
    parent_main_labels: Sequence[str], parent_selected_labels: Sequence[str] = (),
    existing_children: Sequence[dict] = (),
) -> AnnotationResult:
    parent_scope = HalfOpenRange(parent_start, parent_end)
    context = EvaluationContext.from_dataframe(df)
    existing = {
        (int(child.get("start_index", -1)), int(child.get("end_index", -1)), label)
        for child in existing_children
        for label in child.get("labels", [])
    }
    eligible_parents = set(parent_main_labels)
    selected_on_parent = set(parent_selected_labels)
    sub_requirements = skills.get(
        "sub_label_annotation.sub_label_selection_requirements", {},
    )
    parent_children = [
        label_id
        for parent in eligible_parents
        for label_id in LABEL_CATEGORIES.get(parent, [])
        if label_id in sub_requirements and label_id not in selected_on_parent
    ]
    segment_type_requirements = skills.get(
        "sub_label_annotation.segment_type_selection_requirements", {},
    )
    segment_types = list(segment_type_requirements)
    raw_annotations: List[dict] = []
    conflicts: List[Tuple[str, str]] = []
    evaluated = evaluate_labels(parent_children, context, parent_scope)
    conflicts.extend(evaluated.conflicts)
    for label_id in evaluated.labels:
        for branch in evaluated.evaluations[label_id].matched_branches:
            range_ = branch.evidence_range
            if range_ is None or not parent_scope.contains(range_):
                continue
            if (range_.start, range_.end, label_id) in existing:
                continue
            raw_annotations.append({
                "label_id": label_id,
                "start_index": range_.start,
                "end_index": range_.end,
                "passed": _passed_with_evidence(branch),
            })
    annotations = _merge_label_annotations(raw_annotations)
    found_ranges = sorted({
        HalfOpenRange(annotation["start_index"], annotation["end_index"])
        for annotation in annotations
    })
    if found_ranges:
        evaluated_types = evaluate_labels(segment_types, context, parent_scope)
        conflicts.extend(evaluated_types.conflicts)
    for range_ in found_ranges:
        for label_id in evaluated_types.labels:
            evaluation = evaluated_types.evaluations[label_id]
            fitting_branch = next(
                (
                    branch for branch in evaluation.matched_branches
                    if branch.evidence_range is not None
                    and branch.evidence_range.contains(range_)
                ),
                None,
            )
            if fitting_branch is None:
                continue
            annotations.append({
                "label_id": label_id,
                "start_index": range_.start,
                "end_index": range_.end,
                "reasoning": _detailed_reason_from_passed(
                    label_id,
                    _passed_with_evidence(fitting_branch),
                    range_,
                ),
            })
    annotations.sort(key=lambda item: (
        int(item["start_index"]), int(item["end_index"]), item["label_id"],
    ))
    labels = list(dict.fromkeys(annotation["label_id"] for annotation in annotations))
    summary = f"Deterministically selected {len(annotations)} label proposal(s)."
    if conflicts:
        summary += f" Suppressed {len(set(conflicts))} exclusive conflict(s)."
    return AnnotationResult(
        final_labels=labels,
        final_reasoning=summary,
        accepted=True,
        iterations=1,
        messages=[],
        label_annotations=annotations,
    )


__all__ = [
    "FACT_REGISTRY", "INPUT_REGISTRY", "SUPPORTED_OPERATORS",
    "calculate_detailed_annotation", "calculate_lap_annotation", "evaluate_labels",
    "evaluate_requirements", "validate_catalog",
]
