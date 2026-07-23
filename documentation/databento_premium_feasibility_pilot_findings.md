# Databento Premium-Feasibility Pilot Findings

## Status

The historical data path is technically viable, but one Friday is not enough
to approve or reject a defensive-strike rule.

## Pilot Date

- Entry date: 2026-05-08
- Decision time: 3:45 p.m. America/New_York
- Underlying future: MNQM6
- Comparable expiration: 2026-06-05, 28 calendar DTE
- Spread width: 150 points

The 35-DTE expiration was excluded because the defensive candidate could not
form an exact 150-point-wide listed spread.

## Executable Structures

| Candidate | Theoretical short target | Executable spread |
|---|---:|---:|
| Standard | 26,165.32 | 25,900 / 25,750 |
| Defensive | 25,288.27 | 25,250 / 25,100 |

The executable short strikes were 650 points apart, approximately 2.22% of the
MNQM6 quote at the decision time. The listed-strike lattice therefore produced
less than the intended 3% separation.

## Quote Availability

All four option legs and MNQM6 had valid BBO-1s observations at or before the
decision time. Option quote ages ranged from 6 to 29 seconds.

## Spread Credits

| Candidate | Natural credit | Midpoint credit | Optimistic credit |
|---|---:|---:|---:|
| Standard | -19.50 | 5.75 | 31.00 |
| Defensive | -20.25 | 3.50 | 27.25 |

Neither spread met the 8.25-point credit floor at midpoint. Crossing the
individual-leg markets would have produced a debit because the outright option
markets were extremely wide.

The defensive midpoint credit haircut was 2.25 points, below the 2.91-point
maximum tolerable haircut from the forward-test expiration proxy.

## Interpretation

The relative economics were supportive: the observed defensive midpoint
haircut was smaller than the modeled expiration-loss benefit.

The absolute economics were not supportive on this date: neither candidate met
the strategy's credit floor at midpoint.

This date would therefore be classified as **no qualifying trade**, not as
evidence that the defensive rule should be adopted.

## UDS and Package-Book Investigation

The saved point-in-time definition response used DBN version 1 and did not
contain component-leg fields for multi-leg instruments. Exact historical
vertical-spread matching could not be performed from that response.

Several later diagnostics identified user-defined outright ES options rather
than MNQ multi-leg spreads. Those outputs were exploratory false leads and
should not be treated as research evidence.

## Next Evidence Required

Run the same point-in-time definition and BBO process on additional
elevated-regime Fridays. The next pilot should measure:

1. Frequency with which both standard and defensive 150-point spreads exist
2. Frequency with which each midpoint credit meets 8.25 points
3. Defensive credit haircut when both structures exist
4. Listed-strike separation actually achieved
5. Quote completeness, age, and width
6. Results across different VIX tiers and stress episodes

The trading rules remain unchanged.
