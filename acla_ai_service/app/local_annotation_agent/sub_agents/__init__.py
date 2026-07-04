"""Plan-step executors registered with the framework on import.

These are capabilities Agent topologies can delegate to:

  * ``label_verifier`` — embedding-similarity filter over the parent's
    candidate labels. Same module exports ``compute_verified_labels`` for
    the local runner to wire as a VLM-callable tool.
"""

# Side-effect imports for non-visual agent capabilities.
from app.local_annotation_agent.sub_agents import label_verifier     # noqa: F401
