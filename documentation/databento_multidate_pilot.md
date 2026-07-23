# Databento Multi-Date Premium Pilot

## Purpose

Extend the May 8 premium-feasibility pilot to two additional elevated-regime
Fridays:

- 2026-04-10
- 2026-04-17

Both dates had regime score 2 in the existing research.

## Current Checkpoint

This checkpoint estimates one UTC day of `ALL_SYMBOLS` instrument definitions
for each Friday. It makes metadata calls only and downloads no historical data.

The definitions are needed because exact listed strikes, expirations, raw
symbols, and underlying futures contracts must be determined point in time
before quote requests can be constructed.

## Cost Control

The next downloader will require an explicit hard maximum total cost. No
definition purchase should be made until the estimates from this checkpoint
have been reviewed.

## Methodological Caution

The configured strike targets come from the existing NDX proxy series. The
eventual quote study must also record the actual MNQ futures price at 3:45 p.m.
and quantify any basis difference before the method is scaled to full history.
