"""Timestamp formatting utilities for lineage tracking in ML service configurations."""
from datetime import UTC, datetime


def utc_timestamp():
    return (
        datetime.now(UTC)
        .replace(microsecond=0)
        .strftime("%Y-%m-%dT%H:%M:%SZ")
    )

def add_timestamp(data: dict, lineage_key: str) -> dict:
    if lineage_key not in data:
        raise ValueError(f"Missing '{lineage_key}' key in config data for lineage tracking.")
    data[lineage_key]["created_at"] = utc_timestamp()
    return data
