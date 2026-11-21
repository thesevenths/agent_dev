"""ARR-RSI-01 P0 -- evidence-grounded recursive research improvement loop."""

from .diagnosis import Diagnosis, DiagnosisOutput, EvidenceGroundedDiagnoser
from .evaluation import Preregistration
from .runner import run_pilot
from .tasks import PublicTask, build_tasks

__all__ = [
    "Diagnosis", "DiagnosisOutput", "EvidenceGroundedDiagnoser",
    "Preregistration", "PublicTask", "build_tasks", "run_pilot",
]
