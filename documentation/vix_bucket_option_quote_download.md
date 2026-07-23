# VIX-Bucket Exact Option-Quote Download

## Cost

The exact two-leg BBO estimate is $0.042297. The recommended command uses a
$0.05 hard ceiling.

September 4, 2020 has an estimate of zero records and zero cost. The downloader
does not submit an empty historical request for that date. It records the date
as `metadata_zero_records`.

## Calculations

For every date with both legs available:

- Natural credit = short bid minus long ask
- Midpoint credit = short midpoint minus long midpoint
- Optimistic credit = short ask minus long bid
- Midpoint dollar credit = midpoint points × $2
- Credit-floor pass = midpoint credit at least 8.25 points

The output also records quote age, timestamp separation between legs, and the
full natural-to-optimistic quote range.

## Interpretation

The outright option BBO can be very wide. Midpoint credit is useful for
comparing regimes, but it is not proof that a spread order would fill at that
price.

This remains a small mapping pilot rather than a final VIX-to-credit formula.
