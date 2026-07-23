# Premium Feasibility Data Contract

## Purpose

Determine whether the proposed defensive strike can be executed at an
economically acceptable credit on elevated-regime Fridays.

The current research has established that the defensive strike reduces proxy
expiration losses. It has **not** established that the farther strike offers
sufficient real-world premium.

This data contract freezes the required evidence before choosing a vendor or
writing a provider-specific downloader.

## Primary Question

On Fridays when the full three-factor score is at least two:

> How often can a 150-point-wide MNQ put credit spread approximately 3% farther
> below spot than the standard VIX-ladder strike be opened at an acceptable
> credit?

## Candidate Definitions

### Standard Candidate

- Entry snapshot: approximately 3:45 p.m. America/New_York on Friday
- Expiration: PM-settled, 26–35 calendar days to expiration
- Theoretical short target: existing VIX-ladder proxy strike
- Listed short strike: nearest available strike at or below the target
- Long strike: 150 points below the listed short strike

### Defensive Candidate

- Same entry snapshot, expiration, settlement type, and spread width
- Theoretical short target: standard theoretical target minus 3% of Friday spot
- Listed short strike: nearest available strike at or below the defensive target
- Long strike: 150 points below the listed defensive short strike

## Row Grain

Each row represents one candidate spread for one:

- Friday entry date
- Expiration
- Candidate type: `standard` or `defensive`

The row contains quotes for both legs of that spread.

## Required Fields

| Field | Meaning |
|---|---|
| `entry_date` | Friday strategy-entry date |
| `quote_timestamp_utc` | Timestamp of the leg quotes in UTC |
| `data_source` | Vendor or source identifier |
| `underlying_symbol` | Underlying futures symbol or continuous identifier |
| `underlying_price` | Underlying value at the quote timestamp |
| `expiration_date` | Option expiration date |
| `dte_calendar` | Calendar days from entry to expiration |
| `settlement_type` | Must identify PM settlement |
| `candidate_type` | `standard` or `defensive` |
| `target_short_strike` | Theoretical short-strike target |
| `actual_short_strike` | Selected listed short strike |
| `actual_long_strike` | Selected listed long strike |
| `short_bid`, `short_ask` | Short-put quote |
| `long_bid`, `long_ask` | Long-put quote |
| `short_open_interest`, `long_open_interest` | Leg open interest |
| `short_volume`, `long_volume` | Leg volume |

## Credit Measures

The validator calculates three credits:

- **Natural credit:** short bid minus long ask
- **Mid credit:** short midpoint minus long midpoint
- **Optimistic credit:** short ask minus long bid

The natural credit is the conservative immediately marketable estimate. The
mid credit is useful for feasibility analysis but is not assumed to be filled.

## Acceptance Checks

The validator requires:

- Quote timestamp within five minutes of 3:45 p.m. New York time
- Quote local date equal to the Friday entry date
- PM settlement
- 26–35 calendar DTE
- Exactly 150-point spread width
- Short strike at or below its theoretical target
- Nonnegative quotes, volume, and open interest
- Bid no greater than ask
- No duplicate candidate rows

## Analysis Outputs Once Data Exists

The subsequent feasibility analysis will report:

1. Percentage of elevated Fridays with usable standard and defensive quotes
2. Natural- and mid-credit availability
3. Percentage meeting the 8.25-point floor
4. Defensive credit haircut relative to the standard spread
5. Comparison with the maximum tolerable haircut from proxy P&L research
6. Liquidity and bid/ask quality
7. Results by VIX tier, year, and market-stress episode
8. Whether the defensive rule should be approved, revised, or rejected

## Decision Discipline

The strategy will not be changed merely because the farther strike reduces
historical intrinsic loss.

A defensive rule requires evidence that:

- Historical quotes are sufficiently complete and reliable
- The defensive spread is available often enough to matter
- Its credit haircut is generally below the tolerable haircut established by
  the proxy analysis
- The result is not confined to one vendor artifact or isolated event
- The operational credit and liquidity requirements can be written
  mechanically

## Provider Independence

This contract does not assume Databento, ThetaData, CME DataMine, or another
vendor. Provider-specific extraction logic should translate source data into
this stable format rather than changing the research question to suit the
vendor.
