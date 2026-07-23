# MNQ Underlying Quote Estimate

## Current Sample

Twelve of the thirteen selected Fridays have:

- an MNQ put expiration between 26 and 35 calendar DTE, and
- at least one exact 150-point strike pair.

October 7, 2022 is excluded because no MNQ put expiration existed inside the
required DTE window.

For this mapping pilot, the remaining twelve dates are sufficient to continue:

- two samples in five buckets,
- one sample in the 30–35 bucket, and
- one sample above VIX 40.

## Expiration Rule

For each usable date, choose the expiration closest to 28 calendar DTE. Ties
are resolved by the lower DTE and then the earlier expiration.

This keeps the selection deterministic while retaining the established
26–35 DTE rule.

## Current Checkpoint

The script estimates ten minutes of BBO-1s data around 3:45 p.m. for only the
selected MNQ futures contract on each date. It downloads nothing.

Those futures midpoints will determine the actual VIX-ladder strike target.
Only after the target is known will exact option legs be selected.
