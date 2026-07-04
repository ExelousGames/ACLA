"""Annotation-domain agents registered with the agent box.

Side-effect imports trigger ``.register()`` on each Agent subclass,
populating the box's AGENT_REGISTRY. The flow modules (detailed, lap)
reference these by name when they build their AgentRequest.

``label_verifier`` lives in ``agent/sub_agents/`` and is registered there
on import of ``run_local``.
"""

from app.local_annotation_agent.workflow.agents import annotation_root  # noqa: F401
