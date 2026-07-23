# Databento MNQ Definition Discovery

## Why This Step Exists

The guessed parent symbols did not resolve. That does not establish that MNQ
options are unavailable. CME options can use multiple product roots for
quarterly, serial, weekly, and daily expirations.

Databento's documented fallback is to request a point-in-time `definition`
snapshot for `ALL_SYMBOLS`, then filter locally using:

- `security_type == "OOF"`
- the actual futures contract in `underlying`
- put/call classification in `instrument_class`
- expiration and strike fields

## Two-Step Cost Control

First run `--estimate`. This calls `metadata.get_cost` and downloads no
historical time-series data.

The later `--download` action requires an explicit `--max-cost-usd` value. It
will refuse to download when the current estimate exceeds that guard.

## Outputs

The raw DBN definition snapshot is stored under `data/raw/databento`, which is
ignored by Git.

Filtered candidate definitions and a summary are written under
`results/databento_definition_discovery`.
