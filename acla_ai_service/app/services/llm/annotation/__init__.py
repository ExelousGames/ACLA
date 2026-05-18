"""
Annotation harness — wraps the agent box with racing-specific intent.

This package sits OUTSIDE the box. It owns:

- Flow definitions (planner/synth prompt builders, raw-text parsers).
- Result dataclasses (AnnotationResult, LapAnnotationResult).
- Domain agents (annotation_root). label_verifier now sits in
  ``agent/sub_agents/`` as a peer of describe_graphs/zoom.
- Domain tools (list_eligible_labels).
- The public ``run_annotation`` entry point UIs call.

The agent box (``app.services.llm.agent``) sees only AgentRequest /
AgentResponse — no AnnotationResult, no skill queries, no flow names.

Side-effect imports register the annotation-domain agents with the box
on import so callers don't need to manage registration order.
"""

# Side-effect imports: register annotation-domain agents, tools, and
# structured-attachment formatters with the agent box.
from app.services.llm.annotation import formatters  # noqa: F401
from app.services.llm.annotation import agents      # noqa: F401
from app.services.llm.annotation import tools       # noqa: F401
