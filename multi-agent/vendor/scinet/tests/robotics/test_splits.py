import unittest

from robotics.data.splits import (
    SPLITS,
    derive_split_seed,
    episode_reset_seed,
    plan_collection,
)

CONFIG = (
    {"random": 100, "guided": 100, "mixture": 100},
    {"train": 0.80, "validation": 0.10, "frozen_test": 0.05, "ood_test": 0.05},
    20261110,
)


class SplitTests(unittest.TestCase):
    def test_derive_split_seed_deterministic_and_distinct(self):
        s1 = derive_split_seed(123, "train")
        s2 = derive_split_seed(123, "train")
        self.assertEqual(s1, s2)
        self.assertNotEqual(derive_split_seed(123, "train"),
                             derive_split_seed(123, "validation"))
        self.assertNotEqual(derive_split_seed(123, "train"),
                             derive_split_seed(123, "frozen_test"))
        self.assertNotEqual(derive_split_seed(123, "train"),
                             derive_split_seed(123, "ood_test"))

    def test_plan_collection_counts(self):
        plan = plan_collection(*CONFIG)
        self.assertEqual(len(plan), 300)
        by_split, by_policy = {}, {}
        for it in plan:
            by_split[it["split"]] = by_split.get(it["split"], 0) + 1
            by_policy[it["policy"]] = by_policy.get(it["policy"], 0) + 1
        self.assertEqual(by_policy, {"random": 100, "guided": 100, "mixture": 100})
        self.assertEqual(by_split["train"], 240)
        self.assertEqual(by_split["validation"], 30)
        self.assertEqual(by_split["frozen_test"], 15)
        self.assertEqual(by_split["ood_test"], 15)

    def test_plan_episode_ids_unique_no_overlap(self):
        plan = plan_collection({"random": 50, "guided": 50},
                               {"train": 0.5, "validation": 0.25,
                                "frozen_test": 0.25, "ood_test": 0},
                               7)
        ids = [it["episode_id"] for it in plan]
        self.assertEqual(len(ids), len(set(ids)))
        split_ids = {}
        for it in plan:
            split_ids.setdefault(it["split"], set()).add(it["episode_id"])
        keys = list(split_ids)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                self.assertEqual(split_ids[keys[i]] & split_ids[keys[j]], set())

    def test_same_config_same_plan(self):
        p1 = plan_collection(*CONFIG)
        p2 = plan_collection(*CONFIG)
        self.assertEqual(
            [(it["episode_id"], it["reset_seed"]) for it in p1],
            [(it["episode_id"], it["reset_seed"]) for it in p2],
        )

    def test_reset_seed_is_split_local(self):
        self.assertNotEqual(episode_reset_seed(1, "train", 0),
                             episode_reset_seed(1, "validation", 0))

    def test_all_splits_present_in_constant(self):
        self.assertEqual(SPLITS, ("train", "validation", "frozen_test", "ood_test"))


if __name__ == "__main__":
    unittest.main()
