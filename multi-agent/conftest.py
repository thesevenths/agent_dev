"""Repo-root conftest.

Guarantees the repository root is importable so package-qualified test imports
(``from tests.rt_tests.research_fakes import ...``) resolve regardless of how
pytest is invoked (CI runner, ``python -m pytest``, IDE).
"""
