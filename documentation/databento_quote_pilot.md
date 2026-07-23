# Databento Selected-Leg Quote Pilot

## Joint Executability Rule

A candidate expiration is included only when both structures exist:

- Standard 150-point-wide spread
- Defensive 150-point-wide spread

The nearest short strike may not have a listed long strike exactly 150 points
below it. The script therefore searches downward through listed short strikes
until it finds an exact-width pair.

If either candidate is unavailable, that expiration is excluded and recorded
in `spread_selection_audit.csv`. This is an operational feasibility result, not
a software error.

## Why Both Candidates Must Exist

The premium study compares the standard and defensive structures on the same
Friday and expiration. Using different expirations would mix the strike effect
with different time-to-expiration and term-structure effects.

## Cost Estimate

Only the exact legs from jointly executable expirations, plus the underlying
future, are submitted to Databento metadata endpoints. No historical quote data
is downloaded.
