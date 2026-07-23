# Pure-MNQ Definition Download

## Why Definitions Come Before Strike Selection

The VIX ladder must be applied to the actual MNQ futures price at 3:45 p.m.,
not to NDX.

The definition snapshots are therefore used only to establish:

- the MNQ futures contract associated with each option series,
- every listed put strike,
- every 26–35 DTE expiration, and
- whether exact 150-point strike pairs exist.

No short strike is selected in this checkpoint.

## Next Checkpoint

The next request will retrieve only the MNQ futures quote around 3:45 p.m. for
each selected Friday. Those prices will determine the ladder targets.

Afterward, the locally saved definition inventories will identify the closest
executable 150-point spreads. Only those exact option legs will be purchased in
the final, inexpensive quote request.

## Cost Control

The estimated definition cost is $5.502289. The recommended command uses a
$5.60 hard ceiling and reuses any DBN file that already exists.
