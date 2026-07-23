"""Inspect Databento DBN version and definition-schema capabilities.

This diagnostic makes no historical time-series request and incurs no data
charge. It reports:

- local Databento Python package version
- local raw definition DBN metadata version
- whether the saved file exposes strategy-leg fields
- current metadata field list for the definition schema
- Historical client and get_range signatures
- available upgrade-policy classes and enum values

Run:
    python scripts/inspect_databento_dbn_version.py
"""

from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFINITION_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "databento"
    / "definitions"
)


def load_api_key() -> str:
    """Load the API key without displaying it."""

    load_dotenv(
        PROJECT_ROOT / ".env",
        override=True,
    )

    key = os.getenv(
        "DATABENTO_API_KEY",
        "",
    ).strip()

    if not key:
        raise RuntimeError(
            "DATABENTO_API_KEY is missing from .env."
        )

    return key


def find_definition_file() -> Path:
    """Find the most recent local definition DBN file."""

    files = sorted(
        DEFINITION_DIR.glob(
            "*definitions.dbn.zst"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not files:
        raise FileNotFoundError(
            f"No definition DBN file found under {DEFINITION_DIR}"
        )

    return files[0]


def safe_signature(value: Any) -> str:
    """Return a callable signature or a readable failure."""

    try:
        return str(
            inspect.signature(value)
        )
    except Exception as exc:
        return (
            f"signature unavailable: "
            f"{type(exc).__name__}: {exc}"
        )


def metadata_value(
    metadata: Any,
    name: str,
) -> Any:
    """Read a metadata attribute without assuming dict/object form."""

    if isinstance(metadata, dict):
        return metadata.get(name)

    return getattr(
        metadata,
        name,
        None,
    )


def extract_field_name(field: Any) -> str:
    """Extract a field name from metadata.list_fields output."""

    if isinstance(field, dict):
        for key in (
            "name",
            "field",
        ):
            if key in field:
                return str(field[key])

    for attribute in (
        "name",
        "field",
    ):
        value = getattr(
            field,
            attribute,
            None,
        )

        if value is not None:
            return str(value)

    return str(field)


def main() -> int:
    """Run the no-cost diagnostic."""

    print(
        "MNQ PCS Regime Research — Databento DBN Diagnostic"
    )
    print("=" * 58)

    try:
        import databento as db

        key = load_api_key()
        definition_path = (
            find_definition_file()
        )

        print(
            f"Databento package version: "
            f"{getattr(db, '__version__', 'unknown')}"
        )

        print(
            f"Definition file: {definition_path}"
        )

        store = db.DBNStore.from_file(
            definition_path
        )

        metadata = store.metadata

        print("\nSaved DBN metadata")
        print("-" * 84)

        for name in (
            "version",
            "dataset",
            "schema",
            "start",
            "end",
            "stype_in",
            "stype_out",
        ):
            print(
                f"{name:<12}: "
                f"{metadata_value(metadata, name)}"
            )

        frame = store.to_df(
            price_type="float",
            pretty_ts=True,
        ).reset_index()

        leg_columns = sorted(
            column
            for column in frame.columns
            if column.startswith("leg_")
        )

        print("\nSaved definition DataFrame")
        print("-" * 84)

        print(
            f"Rows: {len(frame):,}"
        )

        print(
            f"Columns: {len(frame.columns):,}"
        )

        print(
            "Strategy-leg columns: "
            + (
                ", ".join(leg_columns)
                if leg_columns
                else "NONE"
            )
        )

        print(
            f"Has leg_count: "
            f"{'leg_count' in frame.columns}"
        )

        print(
            f"Has leg_raw_symbol: "
            f"{'leg_raw_symbol' in frame.columns}"
        )

        client = db.Historical(key)

        print("\nClient signatures")
        print("-" * 84)

        print(
            "Historical"
            f"{safe_signature(db.Historical)}"
        )

        print(
            "timeseries.get_range"
            f"{safe_signature(client.timeseries.get_range)}"
        )

        print(
            "DBNStore.request_full_definitions"
            f"{safe_signature(store.request_full_definitions)}"
        )

        print("\nUpgrade-related objects")
        print("-" * 84)

        upgrade_names = sorted(
            name
            for name in dir(db)
            if "upgrade" in name.lower()
        )

        if not upgrade_names:
            print(
                "No top-level upgrade-related objects found."
            )
        else:
            for name in upgrade_names:
                value = getattr(
                    db,
                    name,
                )

                print(
                    f"{name}: {value}"
                )

                members = getattr(
                    value,
                    "__members__",
                    None,
                )

                if members:
                    print(
                        "  members: "
                        + ", ".join(
                            f"{key}={member}"
                            for key, member
                            in members.items()
                        )
                    )

        print("\nCurrent definition field metadata")
        print("-" * 84)

        list_fields_signature = safe_signature(
            client.metadata.list_fields
        )

        print(
            "metadata.list_fields"
            f"{list_fields_signature}"
        )

        fields = None
        field_error = None

        attempts = (
            {
                "encoding": "dbn",
                "schema": "definition",
            },
            {
                "schema": "definition",
                "encoding": "dbn",
            },
            {
                "schema": "definition",
            },
        )

        for arguments in attempts:
            try:
                fields = (
                    client.metadata.list_fields(
                        **arguments
                    )
                )
                break
            except TypeError as exc:
                field_error = exc
                continue

        if fields is None:
            print(
                "Unable to call metadata.list_fields: "
                f"{field_error}"
            )
        else:
            field_names = sorted(
                {
                    extract_field_name(field)
                    for field in fields
                }
            )

            current_leg_fields = [
                name
                for name in field_names
                if name.startswith("leg_")
            ]

            print(
                f"Current field count: "
                f"{len(field_names)}"
            )

            print(
                "Current strategy-leg fields: "
                + (
                    ", ".join(
                        current_leg_fields
                    )
                    if current_leg_fields
                    else "NONE"
                )
            )

        print("\nDiagnostic conclusion")
        print("-" * 84)

        if leg_columns:
            print(
                "The saved definition file already contains "
                "strategy-leg fields. The UDS parser needs correction."
            )
        else:
            print(
                "The saved definition file does not contain "
                "strategy-leg fields."
            )

            print(
                "Use the metadata version, client signature, and "
                "upgrade-policy output above to choose the correct "
                "version-aware re-request."
            )

        print(
            "\nNo historical time-series request was made."
        )

        return 0

    except Exception as exc:
        print(
            f"\nERROR: {type(exc).__name__}: {exc}"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
