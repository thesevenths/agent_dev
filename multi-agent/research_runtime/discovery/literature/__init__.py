"""M7 literature retrieval: real API providers, query expansion, dedup, coverage."""
from .base import (LiteratureError, LiteratureProvider, Paper, ResearchQuestionInput,
                   SearchQuery, deduplicate, normalize_title, paper_identity)
from .providers import (CrossrefProvider, OpenAlexProvider,
                        SemanticScholarProvider, PROVIDERS)
from .store import LiteratureStore, expand_query

__all__ = [n for n in dir() if not n.startswith('_')]
