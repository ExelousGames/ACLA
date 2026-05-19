"""Annotation-domain agents registered with the agent box.

Side-effect imports trigger ``.register()`` on each Agent subclass,
populating the box's AGENT_REGISTRY. The flow modules (detailed, lap)
reference these by name when they build their AgentRequest.

``label_verifier`` is now a peer of ``describe_graphs`` and ``zoom`` in
``agent/sub_agents/`` — registered there on import of ``run_local``.
"""

from app.pipelines.annotation.agents import annotation_root  # noqa: F401
