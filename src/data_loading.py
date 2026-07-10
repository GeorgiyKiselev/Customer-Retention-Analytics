"""CSV discovery, schema normalization, and transaction-table loading."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

LOGGER = logging.getLogger(__name__)

# Edit values in this dictionary when a dataset uses names that auto-detection
# cannot recognize. Values are normalized CSV column names (lowercase snake_case).
# Set "file" to either a CSV filename or a path relative to --data-dir.
# Leave values as None to use auto-detection.
MANUAL_MAPPINGS: dict[str, dict[str, str | None]] = {
    "transactions": {
        "file": None,
        "customer_id": None,
        "order_id": None,
        "order_date": None,
        "amount": None,
        "product": None,
        "category": None,
        "order_status": None,
    },
    "customers": {
        "file": None,
        "customer_id": None,
    },
}

FIELD_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "customer_id": (
        "customer_id", "customerid", "cust_id", "customer_key", "customer_number",
        "client_id", "buyer_id", "user_id",
    ),
    "order_id": (
        "order_id", "orderid", "transaction_id", "transactionid", "invoice_no",
        "invoice_id", "sales_order_id", "purchase_id",
    ),
    "order_date": (
        "order_date", "transaction_date", "purchase_date", "invoice_date",
        "sales_date", "date", "full_date", "date_id",
    ),
    "amount": (
        "net_amount", "revenue", "sales", "sales_amount", "order_amount",
        "transaction_amount", "total_amount", "monetary_value", "gross_amount",
        "amount", "price",
    ),
    "product": (
        "product_id", "product_name", "item_id", "item_name", "sku",
    ),
    "category": (
        "category", "product_category", "category_name", "sub_category",
        "subcategory", "department",
    ),
    "order_status": (
        "order_status", "transaction_status", "status",
    ),
}

REQUIRED_FIELDS: Final[tuple[str, ...]] = (
    "customer_id", "order_id", "order_date", "amount",
)


@dataclass(frozen=True)
class CsvSchema:
    """A discovered CSV and its normalized header."""

    path: Path
    columns: tuple[str, ...]
    encoding: str


@dataclass(frozen=True)
class DatasetMapping:
    """Resolved source files and columns used by the pipeline."""

    transaction_file: Path
    transaction_columns: dict[str, str]
    transaction_encoding: str = "utf-8"
    customer_file: Path | None = None
    customer_id_column: str | None = None


def normalize_column_name(value: object) -> str:
    """Convert a column label to lowercase snake_case."""
    text = str(value).strip()
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_").lower()


def discover_csv_schemas(data_dir: Path) -> list[CsvSchema]:
    """Find CSVs recursively and inspect their headers with pandas."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Data folder does not exist: {data_dir}")
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Data path is not a folder: {data_dir}")

    paths = sorted(
        (path for path in data_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".csv"),
        key=lambda path: str(path).lower(),
    )
    if not paths:
        raise ValueError(f"No CSV files were found in data folder: {data_dir}")

    schemas: list[CsvSchema] = []
    for path in paths:
        header: pd.DataFrame | None = None
        errors: list[str] = []
        selected_encoding = "utf-8"
        for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
            try:
                header = pd.read_csv(path, nrows=0, encoding=encoding)
                selected_encoding = encoding
                break
            except UnicodeDecodeError as exc:
                errors.append(f"{encoding}: {exc}")
            except Exception as exc:
                raise ValueError(f"Could not read CSV header from '{path}': {exc}") from exc
        if header is None:
            raise ValueError(
                f"Could not decode CSV header from '{path}'. Attempts: {'; '.join(errors)}"
            )
        normalized = tuple(normalize_column_name(column) for column in header.columns)
        duplicates = sorted({name for name in normalized if normalized.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"Columns in '{path}' collide after normalization: {duplicates}. "
                f"Original columns: {list(header.columns)}"
            )
        schemas.append(
            CsvSchema(path=path, columns=normalized, encoding=selected_encoding)
        )
        LOGGER.info("Discovered CSV: %s", path)
        LOGGER.info("  Columns: %s", ", ".join(normalized))
        if selected_encoding not in ("utf-8", "utf-8-sig"):
            LOGGER.info("  Detected encoding: %s", selected_encoding)
    return schemas


def _manual_file(
    schemas: list[CsvSchema], data_dir: Path, configured: str | None
) -> CsvSchema | None:
    if not configured:
        return None
    desired = (data_dir / configured).resolve()
    for schema in schemas:
        if schema.path.resolve() == desired or schema.path.name.lower() == configured.lower():
            return schema
    raise ValueError(
        f"Configured CSV file '{configured}' was not found. Discovered files: "
        f"{[schema.path.name for schema in schemas]}"
    )


def _match_fields(
    schema: CsvSchema, manual: dict[str, str | None]
) -> dict[str, str]:
    columns = set(schema.columns)
    matches: dict[str, str] = {}
    for field, aliases in FIELD_ALIASES.items():
        override = manual.get(field)
        if override:
            normalized_override = normalize_column_name(override)
            if normalized_override not in columns:
                continue
            matches[field] = normalized_override
            continue
        match = next((alias for alias in aliases if alias in columns), None)
        if match:
            matches[field] = match
    return matches


def resolve_dataset_mapping(
    schemas: list[CsvSchema], data_dir: Path
) -> DatasetMapping:
    """Auto-detect the transaction table and optional customer table."""
    tx_manual = MANUAL_MAPPINGS["transactions"]
    selected = _manual_file(schemas, data_dir, tx_manual["file"])

    candidates: list[tuple[int, CsvSchema, dict[str, str]]] = []
    for schema in schemas:
        matches = _match_fields(schema, tx_manual)
        required_count = sum(field in matches for field in REQUIRED_FIELDS)
        filename_bonus = 3 if any(
            word in schema.path.stem.lower() for word in ("order", "transaction", "sales", "invoice")
        ) else 0
        candidates.append((required_count * 10 + len(matches) + filename_bonus, schema, matches))

    if selected:
        tx_matches = _match_fields(selected, tx_manual)
        tx_schema = selected
    else:
        _, tx_schema, tx_matches = max(candidates, key=lambda item: item[0])

    missing = [field for field in REQUIRED_FIELDS if field not in tx_matches]
    if missing:
        available = "\n".join(
            f"  - {schema.path.name}: {', '.join(schema.columns)}" for schema in schemas
        )
        raise ValueError(
            "Could not map required transaction fields "
            f"{missing}. Available normalized columns:\n{available}\n"
            "Update MANUAL_MAPPINGS in src/data_loading.py with the correct file "
            "and normalized column names."
        )

    customer_manual = MANUAL_MAPPINGS["customers"]
    customer_schema = _manual_file(schemas, data_dir, customer_manual["file"])
    customer_id_column: str | None = None
    if customer_schema:
        customer_id_column = _match_fields(customer_schema, customer_manual).get("customer_id")
    else:
        likely_customers = [
            schema for schema in schemas
            if schema != tx_schema
            and "customer_id" in _match_fields(schema, customer_manual)
            and "customer" in schema.path.stem.lower()
        ]
        if likely_customers:
            # Prefer an explicitly named dimension over derived customer facts
            # such as a precomputed RFM table.
            customer_schema = max(
                likely_customers,
                key=lambda schema: (
                    int(schema.path.stem.lower().startswith(("dim_customer", "customer"))),
                    int("registration_date" in schema.columns),
                    -len(schema.columns),
                ),
            )
            customer_id_column = _match_fields(customer_schema, customer_manual).get("customer_id")

    mapping = DatasetMapping(
        transaction_file=tx_schema.path,
        transaction_columns=tx_matches,
        transaction_encoding=tx_schema.encoding,
        customer_file=customer_schema.path if customer_schema else None,
        customer_id_column=customer_id_column,
    )
    LOGGER.info("Selected transaction table: %s", mapping.transaction_file)
    LOGGER.info("Resolved transaction mapping: %s", mapping.transaction_columns)
    if mapping.customer_file:
        LOGGER.info(
            "Detected optional customer table: %s (customer id: %s)",
            mapping.customer_file,
            mapping.customer_id_column,
        )
    return mapping


def _parse_dates(series: pd.Series, source_name: str) -> pd.Series:
    values = series.astype("string").str.strip()
    compact_date = values.str.fullmatch(r"\d{8}", na=False)
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    if compact_date.any():
        parsed.loc[compact_date] = pd.to_datetime(
            values.loc[compact_date], format="%Y%m%d", errors="coerce"
        )
    if (~compact_date).any():
        try:
            parsed.loc[~compact_date] = pd.to_datetime(
                values.loc[~compact_date], format="mixed", errors="coerce"
            )
        except (TypeError, ValueError):
            parsed.loc[~compact_date] = pd.to_datetime(
                values.loc[~compact_date], errors="coerce"
            )
    invalid = parsed.isna().sum()
    if invalid:
        examples = values.loc[parsed.isna()].dropna().unique()[:5].tolist()
        LOGGER.warning(
            "Dropping %d rows with invalid %s values. Examples: %s",
            invalid, source_name, examples,
        )
    return parsed


def _parse_amounts(series: pd.Series, source_name: str) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        parsed = pd.to_numeric(series, errors="coerce")
    else:
        cleaned = (
            series.astype("string")
            .str.strip()
            .str.replace(r"^\((.*)\)$", r"-\1", regex=True)
            .str.replace(r"[$£€₹,\s]", "", regex=True)
        )
        parsed = pd.to_numeric(cleaned, errors="coerce")
    invalid = parsed.isna().sum()
    if invalid:
        examples = series.loc[parsed.isna()].dropna().unique()[:5].tolist()
        LOGGER.warning(
            "Dropping %d rows with invalid %s values. Examples: %s",
            invalid, source_name, examples,
        )
    return parsed


def load_transactions(mapping: DatasetMapping) -> pd.DataFrame:
    """Load selected transaction columns and return canonical column names."""
    source_columns = list(dict.fromkeys(mapping.transaction_columns.values()))
    try:
        frame = pd.read_csv(
            mapping.transaction_file,
            usecols=source_columns,
            low_memory=False,
            encoding=mapping.transaction_encoding,
        )
    except Exception as exc:
        raise ValueError(
            f"Could not load transaction table '{mapping.transaction_file}': {exc}"
        ) from exc
    frame.columns = [normalize_column_name(column) for column in frame.columns]

    rename = {source: target for target, source in mapping.transaction_columns.items()}
    frame = frame.rename(columns=rename)
    frame["order_date"] = _parse_dates(frame["order_date"], mapping.transaction_columns["order_date"])
    frame["amount"] = _parse_amounts(frame["amount"], mapping.transaction_columns["amount"])
    frame["customer_id"] = frame["customer_id"].astype("string").str.strip()
    frame["order_id"] = frame["order_id"].astype("string").str.strip()

    required_valid = (
        frame["customer_id"].notna()
        & frame["customer_id"].ne("")
        & frame["order_id"].notna()
        & frame["order_id"].ne("")
        & frame["order_date"].notna()
        & frame["amount"].notna()
    )
    dropped = int((~required_valid).sum())
    if dropped:
        LOGGER.warning("Dropping %d rows missing required transaction values.", dropped)
        frame = frame.loc[required_valid].copy()

    if "order_status" in frame.columns:
        cancelled = frame["order_status"].astype("string").str.lower().str.contains(
            r"cancel|void|failed", na=False
        )
        if cancelled.any():
            LOGGER.info("Excluding %d cancelled/void/failed transaction rows.", cancelled.sum())
            frame = frame.loc[~cancelled].copy()

    if frame.empty:
        raise ValueError("No valid transaction rows remain after parsing required fields.")

    LOGGER.info(
        "Loaded %d transaction rows spanning %s through %s.",
        len(frame),
        frame["order_date"].min().date(),
        frame["order_date"].max().date(),
    )
    return frame
