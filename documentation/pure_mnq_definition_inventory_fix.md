# Complete Pure-MNQ Definition Inventory

All 13 raw definition files were downloaded successfully before the prior
script stopped.

The interruption occurred during local filtering because October 7, 2022 had
no MNQ put expiration in the required 26–35 calendar-DTE window. That is a
market-availability result, not a failed or incomplete download.

This replacement inventory script:

- reads the existing raw files,
- records dates without an eligible expiration as ineligible,
- continues through all remaining dates,
- inventories exact 150-point pairs, and
- downloads nothing.

After the complete inventory is available, ineligible dates can be replaced
within the same VIX bucket without silently changing the DTE rule.
