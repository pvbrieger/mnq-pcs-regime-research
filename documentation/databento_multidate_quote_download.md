# Databento Multi-Date BBO Quote Download

## Question Answered

For April 10 and April 17, did the real historical MNQ option chain provide
enough premium for:

- the standard VIX-ladder spread, and
- the farther-out defensive spread?

## Data Purchased

For each date, the script requests only:

- four exact option legs,
- the corresponding MNQ futures contract, and
- ten minutes of BBO-1s data centered on 3:45 p.m. New York time.

The combined estimate was $0.005862. The recommended command uses a one-cent
hard ceiling.

## Calculations

For each spread:

- Natural credit = short bid minus long ask
- Midpoint credit = short midpoint minus long midpoint
- Optimistic credit = short ask minus long bid
- Credit-floor test = at least 8.25 points

The comparison output also records:

- actual MNQ futures midpoint,
- NDX-to-MNQ basis,
- each selected short's percentage below MNQ,
- actual separation between standard and defensive shorts, and
- whether the midpoint haircut is within the 2.91-point proxy tolerance.

## Data-Quality Policy

April 10 is marked `degraded` by Databento. It remains exploratory and cannot
serve as a clean validation date. April 17 is marked `available`.
