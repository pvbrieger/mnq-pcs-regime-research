# Databento DBN Version Diagnostic

The existing definition file lacks the `leg_*` fields required to identify the
components of CME strategy and user-defined spread instruments.

Databento's current definition schema creates one definition record per
strategy leg. Those fields were introduced with DBN version 3.

This diagnostic confirms the saved file's DBN metadata version and inspects the
installed Python client's upgrade-policy interface before another paid
definition request is attempted.

It makes no historical time-series request.
