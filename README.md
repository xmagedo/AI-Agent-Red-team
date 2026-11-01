# AI Agent Security – Multi-Step Tool Attacks

Workspace for a Kaggle competition submission. The submission is a single
`attack.py` defining an `AttackAlgorithm` class that the evaluator runs against
a sandboxed tool-using agent and then replays.

## Layout

- `attack.py` – submission entry point
- `engine/` – search, scoring, and a local stub environment
- `payloads/` – seed inputs
- `tests/` – unit tests

Requires Python 3.11+ and the competition's `aicomp_sdk` package.
