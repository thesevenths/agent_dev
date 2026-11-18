"""ARR v1.0 WP-02: multi-round Research Program orchestration (thin layer)."""
from .program_state import (
    PROGRAM_STATES, ROUND_STATES, ProgramStateError,
    ResearchProgramState, new_round_record)
from .history_index import ResearchHistoryIndex
from .program_driver import ResearchProgramDriver, ProgramBudgetManager
from .discovery_round import DiscoveryRoundAdapter

__all__ = ["PROGRAM_STATES", "ROUND_STATES", "ProgramStateError",
           "ResearchProgramState", "new_round_record", "ResearchHistoryIndex",
           "ResearchProgramDriver", "ProgramBudgetManager",
           "DiscoveryRoundAdapter"]
