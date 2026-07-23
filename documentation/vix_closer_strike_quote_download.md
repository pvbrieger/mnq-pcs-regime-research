# Closer-Strike MNQ Quote Download

## Cost

The exact multi-distance quote estimate is $0.172982. The recommended command
uses a $0.20 hard ceiling.

September 4, 2020 has zero estimated BBO records and is recorded as unavailable.
September 6, 2024 and April 4, 2025 had no executable exact-width spread at any
requested distance, so they are not part of the option request.

## Economic Measurements

For each date and distance:

- Natural credit = short bid minus long ask
- Midpoint credit = short midpoint minus long midpoint
- Optimistic credit = short ask minus long bid
- Gross edge = midpoint credit minus historical mean intrinsic loss
- Conservative edge = midpoint credit minus the upper historical-loss interval

For each closer distance relative to the same date's current ladder spread:

- Incremental credit = closer midpoint minus current midpoint
- Incremental historical loss = closer expected loss minus current expected loss
- Net incremental edge = incremental credit minus incremental historical loss

A positive net incremental edge means the extra midpoint premium exceeded the
additional historical mean expiration loss.

## Interpretation Limits

- The actual-credit sample remains small.
- Midpoint is not a guaranteed fill.
- Some listed strikes are materially farther out than the requested target.
- Historical loss is joined at the requested target distance, which is generally
  conservative when the executable listed strike is farther out.
- No operational rule should change from one pilot alone.
