"""M4.3-02 -- Confirmatory Runner (complete framework, human-gated start).

The confirmatory stage reuses the SAME execution machinery as the pilot
(``PilotRunner``): one ResearchService, one 12-step gate, one registry, one
A/B/C block protocol. What changes is that everything is FROZEN beforehand:

  * the protocol (20 fields, single SHA-256) is loaded from disk and verified;
  * a HUMAN approval bound to exactly that SHA must exist;
  * the block count is the frozen sample size, and the snapshot ids are the
    frozen set -- the runner refuses to execute anything else;
  * human blind review must be COMPLETE for every queued objective, and the
    frozen final split is never touched.

Default behaviour is to REFUSE: ``run()`` raises unless the freeze and the
approval are both in place. Nothing starts automatically.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .confirmatory_protocol_builder import (
    ProtocolBlocked, ProtocolNotApproved, load_approval, load_frozen_protocol,
)
from .experiment_controller import ExperimentController
from .experiment_registry import FrozenProtocol
from .pilot_runner import PilotConfig, PilotRunner, PreflightError, preflight
from .protocol_schema import missing_confirmatory_fields
from .review_queue import ReviewQueue


class ConfirmatoryRunner(PilotRunner):
    """Execute the FROZEN confirmatory protocol (only after human approval)."""

    stage = "confirmatory"

    def __init__(self, *, service: Any, config: PilotConfig,
                 out_dir: str | Path, controller: ExperimentController,
                 protocol: FrozenProtocol, approval: dict[str, Any],
                 review_queue: ReviewQueue | None = None):
        if not protocol.confirmatory:
            raise ProtocolBlocked("the confirmatory runner needs a frozen "
                                  "confirmatory protocol")
        missing = missing_confirmatory_fields(protocol)
        if missing:
            raise ProtocolBlocked(
                f"the confirmatory protocol is incomplete: {missing}")
        if approval.get("protocol_sha256") != protocol.freeze_sha256():
            raise ProtocolNotApproved(
                "the human approval is bound to a different protocol SHA; "
                "refusing to start")
        if int(protocol.sample_size) != int(config.n_blocks):
            raise ProtocolBlocked(
                f"the frozen sample size is {protocol.sample_size} blocks but "
                f"the runner is configured for {config.n_blocks}; the frozen "
                "protocol decides the sample size")
        if config.require_review_complete is not True:
            raise ProtocolBlocked(
                "the confirmatory stage requires COMPLETED human blind review "
                "for every queued objective")
        super().__init__(service=service, config=config, out_dir=out_dir,
                         controller=controller, review_queue=review_queue,
                         protocol=protocol)

    # ---------------------------------------------------------------- gates
    @classmethod
    def from_frozen(cls, *, service: Any, config: PilotConfig,
                    out_dir: str | Path, controller: ExperimentController,
                    protocol_dir: str | Path,
                    review_queue: ReviewQueue | None = None
                    ) -> "ConfirmatoryRunner":
        protocol = load_frozen_protocol(protocol_dir)
        approval = load_approval(protocol_sha256=protocol.freeze_sha256(),
                                 out_dir=protocol_dir)
        return cls(service=service, config=config, out_dir=out_dir,
                   controller=controller, protocol=protocol,
                   approval=approval, review_queue=review_queue)

    def _snapshot_id(self, block_number: int) -> str:
        ids = list(self.protocol.snapshot_ids)
        if not ids:
            raise ProtocolBlocked("the frozen protocol has no snapshot ids")
        index = (block_number - 1) % len(ids)
        return ids[index]

    # ----------------------------------------------------------------- run
    def run(self) -> dict[str, Any]:
        """Execute the frozen design. Refuses without every precondition."""
        if self.controller.state not in ("CONFIRMATORY_AWAITING_APPROVAL",):
            raise ProtocolNotApproved(
                "the experiment controller is not awaiting confirmatory "
                f"approval (state={self.controller.state!r}); confirmatory "
                "runs never start automatically")
        if not self.controller.record.config.get("confirmatory_protocol_sha256"):
            raise ProtocolNotApproved(
                "the controller has no frozen protocol SHA; freeze first")
        if self.controller.record.config["confirmatory_protocol_sha256"] != \
                self.protocol.freeze_sha256():
            raise ProtocolNotApproved(
                "the controller is bound to a DIFFERENT protocol SHA")
        rep = preflight(self.config, service=self.service)
        if not rep.ready:
            raise PreflightError(
                "refusing to run the REAL confirmatory stage: "
                f"{rep.to_dict()}")
        return super().run()


__all__ = ["ConfirmatoryRunner"]
