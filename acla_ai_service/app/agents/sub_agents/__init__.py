"""Plan-step executors registered with the framework on import.

These are capabilities every Agent topology can delegate to:

  * ``describe_graphs`` — render an image set, ask the VLM to pose zoom
    questions, synthesise prose observations from the zoom answers.
  * ``zoom`` — answer the parent's questions over their sub-ranges using
    deterministic queries from the pipeline-query catalog.
  * ``label_verifier`` — embedding-similarity filter over the parent's
    candidate labels. Same module exports ``compute_verified_labels`` for
    the local runner to wire as a VLM-callable tool.
"""

# Side-effect imports: leaves first, then meta-agents.
from app.agents.sub_agents import zoom               # noqa: F401
from app.agents.sub_agents import label_verifier     # noqa: F401
from app.agents.sub_agents import describe_graphs    # noqa: F401
