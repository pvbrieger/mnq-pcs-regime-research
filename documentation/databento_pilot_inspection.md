# Databento Pilot Inspection

## Purpose

Verify Databento authentication, CME dataset availability, MNQ futures
symbology, options-parent symbology, and estimated request costs before
downloading historical market data.

## Cost Discipline

This inspection uses metadata and symbology endpoints and calls
`metadata.get_cost`. It does not call `timeseries.get_range` and therefore does
not download historical time-series data.

The cost estimates for `bbo-1s` and `mbp-1` deliberately cover the entire
resolved option parent for ten minutes. They are conservative reference points.
The later pilot downloader will first retrieve definitions, identify only the
four required option legs, and then request quotes for those selected symbols.

## API-Key Handling

The script reads `DATABENTO_API_KEY` from the project `.env` file. The key is
never printed and `.env` remains excluded from Git.
