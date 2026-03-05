import pytest
from ml.data.config.schemas.processed import ProcessedConfig
from ml.exceptions import ConfigError

pytestmark = pytest.mark.unit


def _data_info_payload() -> dict:
    return {
        "name": "hotel_bookings",
        "version": "v1",
        "output": {
            "path_suffix": "data.parquet",
            "format": "parquet",
            "compression": "snappy",
        },
    }


def _lineage_payload() -> dict:
    return {
        "created_by": "tests",
        "created_at": "2026-03-05T00:00:00",
    }


def test_processed_config_defaults_sensitive_remove_columns() -> None:
    config = ProcessedConfig.model_validate(
        {
            "data": _data_info_payload(),
            "interim_data_version": "v2",
            "lineage": _lineage_payload(),
        }
    )

    assert config.interim_data_version == "v2"
    assert config.remove_columns == ["name", "email", "phone_number", "credit_card"]


def test_processed_config_rejects_invalid_interim_data_version_format() -> None:
    with pytest.raises(ConfigError, match="Invalid interim_data_version"):
        ProcessedConfig.model_validate(
            {
                "data": _data_info_payload(),
                "interim_data_version": "2",
                "lineage": _lineage_payload(),
            }
        )
