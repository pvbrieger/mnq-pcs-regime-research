# Multi-Date Quote Estimate

The April definition snapshots produced two jointly executable comparisons:

- 2026-04-10: May 15 expiration, 35 DTE
- 2026-04-17: May 15 expiration, 28 DTE

Before buying quotes, this checkpoint:

1. Checks Databento's day-level dataset condition.
2. Lists the exact four option symbols and MNQ futures symbol for each date.
3. Estimates ten minutes of BBO-1s data around 3:45 p.m.
4. Downloads nothing.

## April 10 Data Warning

Databento marked April 10 as `degraded`. Databento defines this as data that is
available but may contain missing records or other correctness issues.

Accordingly:

- April 10 can remain an exploratory pilot.
- Missing instruments or missing quotes on April 10 cannot be treated as strong
  evidence of market unavailability.
- April 10 should not count as a clean validation date unless the condition is
  later repaired or independently verified.

April 17 is evaluated separately according to its returned condition.
