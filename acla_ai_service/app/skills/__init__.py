"""Skill surfaces, split by usage into internal and external.

* :mod:`app.skills.internal` — skills used **inside** the service.
  Currently :mod:`app.skills.internal.annotation`: the VLM annotation
  pipeline's skill registry, query DSL, embedder, and label catalog.
  Loads JSON skills via the ``skills`` singleton's ``get`` / ``find`` /
  ``iter`` verbs; hybrid retrieval lives in ``label_search.search``.

* :mod:`app.skills.external` — **user-facing** skills.
  Currently :mod:`app.skills.external.racing_engineer`: the
  racing-engineer's Markdown corpus and its own minimal loader. Direct
  ``label(id)`` / ``feature(name)`` lookups, no query DSL.

Internal and external are independent — neither imports from the other.
"""
