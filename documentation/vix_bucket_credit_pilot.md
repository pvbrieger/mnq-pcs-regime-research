# Pure-MNQ VIX-Bucket Credit Pilot

## Correction

The study now begins on August 31, 2020, when Micro E-mini Nasdaq-100 options
became available. No pre-launch date is eligible.

## Sample

The pilot selects:

- two Fridays in each VIX bucket from below 15 through 35–40; and
- the sole qualifying post-launch Friday above 40.

That produces 13 dates.

For buckets with two samples, the eligible dates are split into an earlier half
and a later half. One date is randomly selected from each half using a fixed
seed. This creates temporal separation without inventing an era split that some
buckets cannot support.

## Eligibility

- Friday observations only
- 2020-08-31 through 2026-07-02
- at least 0.50 VIX points away from ladder boundaries
- Databento condition must be `available`

## Current Checkpoint

The script estimates one UTC day of `ALL_SYMBOLS` definitions for each selected
Friday. It downloads nothing.

After cost review, those definitions will be used to identify the actual MNQ
future, preferred 28-DTE expiration, exact 150-point spread, and exact option
symbols. The later quote request will be comparatively inexpensive.
