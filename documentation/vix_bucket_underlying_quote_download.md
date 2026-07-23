# MNQ Underlying Quotes and Ladder-Spread Selection

## Cost

The estimated total for twelve exact MNQ futures quote requests is $0.714223.
The recommended command uses a $0.75 hard ceiling.

## Selection Logic

For each date:

1. Read the latest valid MNQ BBO at or before 3:45 p.m. New York time.
2. Compute the midpoint.
3. Apply the applicable VIX-ladder percentage to that MNQ midpoint.
4. Find all exact 150-point put-spread pairs in the selected expiration.
5. Choose the highest short strike at or below the ladder target.

This matches the intended rule without using NDX as a substitute.

## Output

The script creates the exact two-leg option manifest required to estimate the
final historical option-quote cost. No option quotes are downloaded here.
