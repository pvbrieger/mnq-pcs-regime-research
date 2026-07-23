# VIX-Bucket Exact Option-Quote Estimate

## Current Status

Ten of the twelve dates with an MNQ futures quote have an executable exact
150-point ladder spread.

Two dates do not:

- September 6, 2024: the 20–25 VIX bucket
- April 4, 2025: the only post-launch Friday above VIX 40

The April 4 result is itself economically relevant: the option chain did not
list an exact 150-point pair far enough below MNQ to execute the stated 15%
ladder rule.

## Current Checkpoint

This script estimates only the exact short and long option symbols for each of
the ten executable dates. It requests ten minutes of BBO-1s data around
3:45 p.m. New York time.

It downloads nothing.

## Final Measurements

After approval and download, each date will report:

- short-leg bid, ask, midpoint, and quote age;
- long-leg bid, ask, midpoint, and quote age;
- natural spread credit;
- midpoint spread credit;
- optimistic spread credit;
- midpoint dollar credit using the MNQ $2 multiplier;
- quote-width diagnostics; and
- whether the midpoint meets the 8.25-point credit floor.
