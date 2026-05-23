"""Skill surfaces.

Two independent subpackages live here:

* :mod:`app.skills.annotation` — the VLM annotation pipeline's skill registry,
  query DSL, embedder, and label catalog. Loads YAML skills via the
  ``skills`` singleton's ``get`` / ``find`` / ``iter`` / ``search`` verbs.

* :mod:`app.skills.racing_engineer` — the racing-engineer's Markdown corpus
  and its own minimal loader (added in Phase 2). Direct ``label(id)`` /
  ``feature(name)`` lookups, no query DSL.

They share only this parent package marker; neither imports from the other.
"""
