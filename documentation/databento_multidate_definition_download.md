# Databento Multi-Date Definition Download

This checkpoint downloads the April 10 and April 17 point-in-time definition
snapshots under an explicit combined cost ceiling.

For each date, it filters to MNQ-related puts with 26–35 calendar DTE and
selects exact 150-point spreads. An expiration enters the later quote manifest
only when both the standard and defensive structures are executable on that
same expiration.

Raw DBN files remain under the Git-ignored raw-data directory. The filtered
definition candidates, feasibility audit, cost audit, and quote manifest are
reproducible project outputs.

The configured targets still come from the NDX proxy. The later quote step must
record the actual MNQ futures price at 3:45 p.m. and quantify the basis
difference before the process is scaled.
