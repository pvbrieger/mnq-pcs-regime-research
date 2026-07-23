# All-Expiration Ladder-Spread Reselection

The first spread-selection pass chose the expiration closest to 28 DTE before
checking whether that expiration listed strikes far enough below MNQ.

That ordering can unnecessarily exclude a date. For example:

- April 4, 2025 also has a 26-DTE expiration.
- April 11, 2025 also has a 35-DTE expiration.

This local-only correction tests every 26–35 DTE expiration associated with the
already quoted MNQ futures contract. It then chooses the feasible expiration
closest to 28 DTE.

No additional data is purchased.
