
"""Benign scenario exploration using the existing SDK-compatible environment."""

from dataclasses import dataclass
from typing import Any, Iterable

from engine.scoring import score_trace


@dataclass
class SearchResult:
    message: str
    severity: int
    predicates: list[str]
    cells: set[str]
    new_cells: set[str]
    gain: int


class SearchEngine:
    """Run predefined scenarios and measure their coverage."""

    def __init__(self, env: Any):
        self.env = env
        self.seen_cells: set[str] = set()
        self.history: list[SearchResult] = []

    def run(self, scenarios: Iterable[str]) -> list[SearchResult]:
        results = []

        for message in scenarios:
            # Save the environment before the experiment.
            checkpoint = self.env.snapshot()

            try:
                # Run one predefined scenario.
                self.env.interact(message)

                # Collect the agent's recorded behavior.
                trace = self.env.export_trace_dict()

                # Score the trace once; derive newly observed cells from it.
                details = score_trace(trace)
                new_cells = details["cells"] - self.seen_cells

                result = SearchResult(
                    message=message,
                    severity=details["severity"],
                    predicates=details["predicates"],
                    cells=details["cells"],
                    new_cells=new_cells,
                    gain=details["severity"] + 2 * len(new_cells),
                )

                # Bank coverage for subsequent scenarios.
                self.seen_cells.update(result.new_cells)
                self.history.append(result)
                results.append(result)

            finally:
                # Undo this interaction, even if an error occurs.
                self.env.restore(checkpoint)

        return results

    def coverage_report(self) -> dict:
        """Return a summary of all scenarios evaluated."""
        return {
            "scenarios_tested": len(self.history),
            "unique_cells": len(self.seen_cells),
            "cells": sorted(self.seen_cells),
            "total_detected_severity": sum(
                result.severity for result in self.history
            ),
        }
