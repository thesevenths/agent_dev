"""P1-04: deterministic scheduling and job identities."""
from __future__ import annotations

import unittest

from research_runtime.campaign import (
    build_schedule,
    make_job_id,
    trial_id_for,
)
from research_runtime.schemas import CandidateExperiment
from tests.rt_tests.campaign_fakes import DATASET_VERSION


def _candidate(name: str) -> CandidateExperiment:
    return CandidateExperiment(
        candidate_id=name, hypothesis_id=f"H_{name}",
        config_overrides={"train.lr": 0.0005}, seeds=[42, 43],
        dataset_version=DATASET_VERSION,
    )


class CampaignSchedulerTests(unittest.TestCase):
    def test_seed_outer_order(self):
        schedule = build_schedule([42, 43], [_candidate("candidate_A"),
                                            _candidate("candidate_B")])
        self.assertEqual(
            [(item.arm_id, item.seed) for item in schedule],
            [("baseline", 42), ("candidate_A", 42), ("candidate_B", 42),
             ("baseline", 43), ("candidate_A", 43), ("candidate_B", 43)],
        )
        self.assertTrue(all(item.arm_type == "baseline"
                            for item in schedule if item.arm_id == "baseline"))

    def test_trial_ids_are_unique_and_stable(self):
        schedule = build_schedule([42, 43], [_candidate("candidate_A")])
        ids = [item.trial_id for item in schedule]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(ids[0], "baseline_seed_42")
        self.assertEqual(ids[1], "candidate_A_seed_42")
        self.assertEqual(trial_id_for("candidate_A", 43), "candidate_A_seed_43")

    def test_baselines_still_scheduled_without_candidates(self):
        schedule = build_schedule([7, 8], [])
        self.assertEqual([(i.arm_type, i.seed) for i in schedule],
                         [("baseline", 7), ("baseline", 8)])
        self.assertTrue(all(not i.overrides for i in schedule))

    def test_job_ids_are_deterministic_and_stage_scoped(self):
        train = make_job_id("campaign_1", "baseline_seed_42", "train")
        again = make_job_id("campaign_1", "baseline_seed_42", "train")
        evaluate = make_job_id("campaign_1", "baseline_seed_42", "evaluate")
        other = make_job_id("campaign_2", "baseline_seed_42", "train")
        self.assertEqual(train, again)
        self.assertNotEqual(train, evaluate)
        self.assertNotEqual(train, other)
        self.assertEqual(len(train), 32)     # uuid5().hex, safe as a filename

    def test_schedule_carries_the_intervention(self):
        schedule = build_schedule([42], [_candidate("candidate_A")])
        self.assertEqual(schedule[1].overrides, {"train.lr": 0.0005})
        self.assertEqual(schedule[1].hypothesis_id, "H_candidate_A")


if __name__ == "__main__":
    unittest.main()
