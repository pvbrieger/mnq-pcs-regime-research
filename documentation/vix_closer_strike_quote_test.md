# Closer-Strike MNQ Quote Test

## Question

Does moving the short strike closer to MNQ improve economic expectancy, or
does the additional historical expiration loss consume the extra premium?

## Distances Tested

For every VIX bucket:

- current ladder distance,
- 1 percentage point closer,
- 2 percentage points closer, and
- 3 percentage points closer.

Examples:

- VIX below 15: 9%, 8%, 7%, and 6%
- VIX 15–20: 10.5%, 9.5%, 8.5%, and 7.5%
- VIX 20–25: 12.5%, 11.5%, 10.5%, and 9.5%
- VIX 25 and above: 15%, 14%, 13%, and 12%

## Same-Expiration Control

Each date uses one expiration for all four distances.

The selected expiration supports the largest number of requested distances,
then is chosen closest to 28 DTE. This prevents the comparison from attributing
a 28-versus-35-DTE premium difference to strike distance.

## Spread Selection

For each target distance:

- compute the target from the actual 3:45 p.m. MNQ midpoint;
- find exact 150-point spread pairs;
- select the highest short strike at or below the target.

## Historical Risk Comparison

Each distance is joined to the existing historical NDX risk surface:

- breach rate,
- full-loss rate,
- average intrinsic loss, and
- yearly block-bootstrap confidence interval.

The selected listed strike is usually slightly farther out than the requested
target. Using the target-distance historical loss is therefore generally a
conservative comparison.

## Final Economic Test

After exact option quotes are downloaded, each date and distance will report:

- natural, midpoint, and optimistic credit;
- midpoint credit in MNQ dollars;
- gross edge = midpoint credit minus historical average intrinsic loss;
- conservative edge = midpoint credit minus the upper historical-loss interval;
- incremental credit gained by moving closer;
- incremental historical loss incurred by moving closer; and
- net incremental advantage or disadvantage.

Midpoint remains an indicative fair-value estimate rather than a guaranteed
fill.
