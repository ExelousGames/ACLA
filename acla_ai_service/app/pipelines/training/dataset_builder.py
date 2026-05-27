"""Step 2 of the racing-engineer training data flow.

Step 1 (annotation pipeline) tags each lap segment with label IDs. This
module is Step 2: it transforms annotated segments into chat-formatted
training rows the LLM trainer consumes.

One **training unit** = either a parent segment with its children, or a
lone segment with no parent. Parent/child is detected from the segment
store via ``AnnotatedSegment.parent_id``; the annotation UI writes one
unit per record after grouping.

Per unit, the human annotator captures TWO completions in the UI:
``completion_critique`` (model's critique-mode response) and
``completion_guide`` (guide-mode response). The builder fans each unit
out into two training rows so the LLM learns to switch modes from the
System instruction.

Input record (one JSONL line, written by the refactored annotation UI)::

    {
      "unit_id": "<parent_seg_id or lone_seg_id>",
      "kind": "parent_with_children" | "isolated",
      "parent_label_ids": ["EA", "brands_hatch", "druids"],
      "children_label_ids": [["MSP1", "MSP9"], ["RM7"]],
      "completion_critique": "...",
      "completion_guide": "...",
      "timestamp": 1234567890.0
    }

For ``kind == "isolated"``, ``children_label_ids`` is ``[]`` and
``parent_label_ids`` carries the lone segment's labels.

Output row (one JSONL line, chat-template ready for HF SFTTrainer)::

    {"messages": [
        {"role": "system",    "content": "critique user, <labels_text>"},
        {"role": "assistant", "content": "<completion_critique>"}
    ]}

Labels are rendered through ``LABEL_MAPPING`` to human-readable names
(the LLM learns from text, not opaque IDs).
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence

from app.domain.labels import LABEL_MAPPING


LOGGER = logging.getLogger(__name__)

MODES: Sequence[str] = ("critique", "guide")


@dataclass(frozen=True)
class TrainingUnit:
    """One annotated training unit pre-fanout."""
    unit_id: str
    kind: str  # "parent_with_children" | "isolated"
    parent_label_ids: List[str]
    children_label_ids: List[List[str]]
    completion_critique: str
    completion_guide: str

    @classmethod
    def from_record(cls, record: dict) -> "TrainingUnit":
        kind = record["kind"]
        if kind not in ("parent_with_children", "isolated"):
            raise ValueError(f"unknown kind: {kind!r}")
        children = record.get("children_label_ids") or []
        if kind == "isolated" and children:
            raise ValueError(
                f"unit {record.get('unit_id')!r}: kind=isolated must "
                f"have empty children_label_ids, got {children!r}"
            )
        return cls(
            unit_id=str(record["unit_id"]),
            kind=kind,
            parent_label_ids=list(record.get("parent_label_ids") or []),
            children_label_ids=[list(c) for c in children],
            completion_critique=str(record.get("completion_critique") or ""),
            completion_guide=str(record.get("completion_guide") or ""),
        )

    def completion(self, mode: str) -> str:
        return self.completion_critique if mode == "critique" else self.completion_guide


def _names(label_ids: Iterable[str]) -> List[str]:
    return [LABEL_MAPPING.get(str(lid), str(lid)) for lid in label_ids]


def render_labels_text(
    parent_label_ids: Sequence[str],
    children_label_ids: Sequence[Sequence[str]],
) -> str:
    """Render a unit's labels as the human-readable string for the System prompt.

    Isolated unit (no children) → flat ``"Name, Name, Name"``.
    Parent unit → ``"parent: A, B, C; child 1: X, Y; child 2: Z"``.
    """
    parent_text = ", ".join(_names(parent_label_ids)) or "(no labels)"
    if not children_label_ids:
        return parent_text
    child_parts = [
        f"child {i + 1}: {', '.join(_names(cids)) or '(no labels)'}"
        for i, cids in enumerate(children_label_ids)
    ]
    return f"parent: {parent_text}; " + "; ".join(child_parts)


def build_training_rows(unit: TrainingUnit) -> List[dict]:
    """Fan one annotated unit into one chat row per mode."""
    labels_text = render_labels_text(
        unit.parent_label_ids, unit.children_label_ids,
    )
    rows: List[dict] = []
    for mode in MODES:
        completion = unit.completion(mode)
        if not completion:
            LOGGER.warning(
                "unit %s missing completion_%s — skipping that row",
                unit.unit_id, mode,
            )
            continue
        rows.append({
            "messages": [
                {"role": "system", "content": f"{mode} user, {labels_text}"},
                {"role": "assistant", "content": completion},
            ],
        })
    return rows


def _read_units(input_path: Path) -> Iterator[TrainingUnit]:
    with input_path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield TrainingUnit.from_record(json.loads(line))
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                raise ValueError(
                    f"{input_path}:{line_no}: invalid annotation record: {exc}"
                ) from exc


def build_dataset(input_path: Path, output_path: Path) -> int:
    """Stream-transform the annotation JSONL into the chat training JSONL.

    Returns the number of chat rows written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    units_read = 0
    with output_path.open("w", encoding="utf-8") as out_fh:
        for unit in _read_units(input_path):
            units_read += 1
            for row in build_training_rows(unit):
                out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows_written += 1
    LOGGER.info(
        "Built %d chat rows from %d annotated units → %s",
        rows_written, units_read, output_path,
    )
    return rows_written


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Transform annotated segments into chat training rows.",
    )
    parser.add_argument("--input", type=Path, required=True,
                        help="Annotation JSONL from the UI.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Destination chat-format JSONL.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    count = build_dataset(args.input, args.output)
    print(f"wrote {count} rows → {args.output}")


if __name__ == "__main__":
    _cli()
