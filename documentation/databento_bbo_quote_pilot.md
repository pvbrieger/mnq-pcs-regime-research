# Databento BBO-1s Quote Pilot

## Execution Rule

For every requested raw symbol, the script selects the latest valid best bid
and ask **at or before** 3:45 p.m. New York time. It rejects quotes more than
five minutes old and never substitutes a quote that occurred after the strategy
decision time.

## Credit Calculations

- Natural credit: short bid minus long ask
- Mid credit: short midpoint minus long midpoint
- Optimistic credit: short ask minus long bid

The 8.25-point credit floor is tested against natural and midpoint credit.

## Defensive Haircut

The pilot compares standard and defensive credits on the same expiration. It
also compares their observed credit haircut with the 2.91-point maximum
tolerable haircut from the forward-test proxy analysis.

## Raw-Data Handling

The DBN file is stored under `data/raw/databento/quotes`, which is ignored by
Git. The script reuses that file on later runs to avoid paying for the same
request twice.

## Scope

One Friday and one jointly executable expiration cannot validate a trading
rule. This pilot only verifies that:

- exact symbols can be retrieved,
- useful top-of-book quotes exist near the decision time,
- spread credits can be calculated reproducibly, and
- the full-history acquisition process is technically viable.
