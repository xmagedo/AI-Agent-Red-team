"""Submission entry point. The evaluator loads `AttackAlgorithm` from this file."""

from __future__ import annotations

from aicomp_sdk.attacks import AttackAlgorithmBase, AttackCandidate, AttackRunConfig
from aicomp_sdk.core.env.api import AttackEnvProtocol


class AttackAlgorithm(AttackAlgorithmBase):
    def run(self, env: AttackEnvProtocol, config: AttackRunConfig) -> list[AttackCandidate]:
        return []
