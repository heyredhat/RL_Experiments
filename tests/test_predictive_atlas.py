"""Tests for weak-beacon localization and predictive-atlas planning."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from predictive_atlas import (
    BEACON_ACTIONS,
    BeaconGRULocalizer,
    beacon_fields,
    collect_localization_dataset,
    exact_transition_joint,
    planning_values,
    update_belief,
)
from spatial_hodology import exact_movement_costs


class PredictiveAtlasTests(unittest.TestCase):
    def test_beacon_fingerprints_overlap_but_distinguish_sites(self) -> None:
        fields = beacon_fields()
        self.assertEqual(fields.shape, (4, 9))
        self.assertTrue(np.all((fields > 0.0) & (fields < 1.0)))
        self.assertEqual(len(np.unique(fields.T, axis=0)), 9)

    def test_delayed_terminal_probe_labels_scan_sequences(self) -> None:
        actions, outcomes, labels = collect_localization_dataset(
            environment="qudit-grid-3x3-beacons",
            cycles=2,
            samples_per_site=2,
            seed=4,
        )
        self.assertEqual(actions.shape, (18, 2 * len(BEACON_ACTIONS)))
        self.assertEqual(outcomes.shape, actions.shape)
        self.assertEqual(sorted(labels.tolist()), sorted(list(range(9)) * 2))
        self.assertTrue(np.all(np.isin(outcomes, (0, 1))))

    def test_recurrent_localizer_accepts_action_outcome_histories(self) -> None:
        model = BeaconGRULocalizer()
        actions = torch.tensor([list(BEACON_ACTIONS) * 2], dtype=torch.long)
        outcomes = torch.zeros_like(actions)
        self.assertEqual(tuple(model(actions, outcomes).shape), (1, 9))

    def test_exact_belief_planner_recovers_stochastic_shortest_paths(self) -> None:
        joint = exact_transition_joint()
        self.assertTrue(np.allclose(joint.sum(axis=(1, 3)), 1.0))
        values, _ = planning_values(joint)
        self.assertTrue(np.allclose(values, exact_movement_costs(0.715), atol=1e-7))

    def test_outcome_conditioned_belief_update_tracks_a_blind_move(self) -> None:
        joint = exact_transition_joint()
        center = np.eye(9)[4]
        north_success = update_belief(center, joint, action=0, outcome=0)
        self.assertEqual(int(np.argmax(north_success)), 1)
        north_again_failure = update_belief(north_success, joint, action=0, outcome=1)
        self.assertEqual(int(np.argmax(north_again_failure)), 1)


if __name__ == "__main__":
    unittest.main()
