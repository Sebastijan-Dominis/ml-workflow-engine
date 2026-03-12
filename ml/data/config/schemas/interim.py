"""Validation schemas for interim-stage data processing configuration."""

import logging
from datetime import datetime

from pydantic import BaseModel, Field, model_validator

from ml.data.config.schemas.constants import BorderValue
from ml.data.config.schemas.shared import DataInfo
from ml.exceptions import ConfigError
from ml.policies.data.interim_constraints import (
    ALLOWED_VALUES_CONSTRAINTS,
    MAX_CONSTRAINTS,
    MIN_CONSTRAINTS,
)

logger = logging.getLogger(__name__)

class DataSchema(BaseModel):
    """Expected interim dataset column names and target dtypes."""

    hotel: str = "category"
    is_canceled: str = "int8"
    lead_time: str = "int16"
    arrival_date_year: str = "int16"
    arrival_date_month: str = "category"
    arrival_date_week_number: str = "int8"
    arrival_date_day_of_month: str = "int8"
    stays_in_weekend_nights: str = "int8"
    stays_in_week_nights: str = "int8"
    adults: str = "int16"
    children: str = "int8"
    babies: str = "int8"
    meal: str = "category"
    country: str = "category"
    market_segment: str = "category"
    distribution_channel: str = "category"
    is_repeated_guest: str = "int8"
    previous_cancellations: str = "int8"
    previous_bookings_not_canceled: str = "int8"
    reserved_room_type: str = "category"
    assigned_room_type: str = "category"
    booking_changes: str = "int8"
    deposit_type: str = "category"
    agent: str = "category"
    company: str = "category"
    days_in_waiting_list: str = "int16"
    customer_type: str = "category"
    adr: str = "float32"
    required_car_parking_spaces: str = "int8"
    total_of_special_requests: str = "int8"
    reservation_status: str = "category"
    reservation_status_date: str = "datetime64[ns]"
    name: str = "string"
    email: str = "string"
    phone_number: str = "string"
    credit_card: str = "string"

class Cleaning(BaseModel):
    """Column-name normalization options for preprocessing input data."""

    lowercase_columns: bool = Field(True, description="Whether to convert column names to lowercase.")
    strip_strings: bool = Field(True, description="Whether to strip leading/trailing whitespace from string columns.")
    replace_spaces_in_columns: bool = Field(True, description="Whether to replace spaces in column names with underscores.")
    replace_dashes_in_columns: bool = Field(True, description="Whether to replace dashes in column names with underscores.")



class Invariant(BaseModel):
    """Validation and filtering rules for a single column."""

    min: BorderValue | None = Field(None, description="Minimum allowed value for the column.")
    max: BorderValue | None = Field(None, description="Maximum allowed value for the column.")
    allowed_values: list | None = Field(None, description="List of allowed values for the column.")

class Invariants(BaseModel):
    """Column-level invariant rules covering the interim dataset schema."""

    hotel: Invariant | None = None
    is_canceled: Invariant | None = None
    lead_time: Invariant | None = None
    arrival_date_year: Invariant | None = None
    arrival_date_month: Invariant | None = None
    arrival_date_week_number: Invariant | None = None
    arrival_date_day_of_month: Invariant | None = None
    stays_in_weekend_nights: Invariant | None = None
    stays_in_week_nights: Invariant | None = None
    adults: Invariant | None = None
    children: Invariant | None = None
    babies: Invariant | None = None
    meal: Invariant | None = None
    country: Invariant | None = None
    market_segment: Invariant | None = None
    distribution_channel: Invariant | None = None
    is_repeated_guest: Invariant | None = None
    previous_cancellations: Invariant | None = None
    previous_bookings_not_canceled: Invariant | None = None
    reserved_room_type: Invariant | None = None
    assigned_room_type: Invariant | None = None
    booking_changes: Invariant | None = None
    deposit_type: Invariant | None = None
    agent: Invariant | None = None
    company: Invariant | None = None
    days_in_waiting_list: Invariant | None = None
    customer_type: Invariant | None = None
    adr: Invariant | None = None
    required_car_parking_spaces: Invariant | None = None
    total_of_special_requests: Invariant | None = None
    reservation_status: Invariant | None = None
    reservation_status_date: Invariant | None = None
    name: Invariant | None = None
    email: Invariant | None = None
    phone_number: Invariant | None = None
    credit_card: Invariant | None = None

    @model_validator(mode="after")
    def validate_constraints(self):
        """Ensure configured invariants stay within registry-defined limits.

        Args:
            self: Candidate invariants instance.

        Returns:
            Invariants: Validated invariants object.
        """
        for field_name, min_allowed in MIN_CONSTRAINTS.items():
            invariant = getattr(self, field_name)
            if invariant and invariant.min is not None and invariant.min.value < min_allowed.value:
                msg = (
                    f"Invalid invariant for '{field_name}': "
                    f"min value {invariant.min.value} is less than {min_allowed.value}."
                )
                logger.error(msg)
                raise ConfigError(msg)

        for field_name, max_allowed in MAX_CONSTRAINTS.items():
            invariant = getattr(self, field_name)
            if invariant and invariant.max is not None and invariant.max.value > max_allowed.value:
                msg = (
                    f"Invalid invariant for '{field_name}': "
                    f"max value {invariant.max.value} is greater than {max_allowed.value}."
                )
                logger.error(msg)
                raise ConfigError(msg)

        for field_name, allowed_values in ALLOWED_VALUES_CONSTRAINTS.items():
            invariant = getattr(self, field_name)
            if invariant and invariant.allowed_values is not None:
                invalid_values = set(invariant.allowed_values) - set(allowed_values)
                if invalid_values:
                    msg = (
                        f"Invalid invariant for '{field_name}': "
                        f"allowed values {invalid_values} are not in the predefined allowed values."
                    )
                    logger.error(msg)
                    raise ConfigError(msg)

        return self

    @model_validator(mode="before")
    # assign default invariants for columns that are not specified in the config
    def assign_default_invariants(cls, values):
        """Populate missing column invariants with registry-based defaults.

        Args:
            values: Raw invariants payload dictionary.

        Returns:
            dict: Invariants payload with defaults assigned.
        """
        for field_name in DataSchema.model_fields:
            if field_name not in values:
                values[field_name] = Invariant(
                    min=MIN_CONSTRAINTS.get(field_name),
                    max=MAX_CONSTRAINTS.get(field_name),
                    allowed_values=ALLOWED_VALUES_CONSTRAINTS.get(field_name)
                )
        return values

class LineageConfig(BaseModel):
    """Lineage metadata describing interim config provenance."""

    created_by: str
    created_at: datetime

class InterimConfig(BaseModel):
    """Top-level validated configuration for interim data creation."""

    data: DataInfo
    data_schema: DataSchema
    raw_data_version: str
    cleaning: Cleaning
    invariants: Invariants
    drop_duplicates: bool = Field(True, description="Whether to drop duplicate rows from the data (default: True).")
    drop_missing_ints: bool = Field(True, description="Whether to drop rows with missing values in integer columns (default: True).")
    min_rows: int = Field(0, description="Minimum number of rows required after cleaning (default: 0).")
    lineage: LineageConfig

    @model_validator(mode="after")
    def validate_raw_data_version(self):
        """Validate that ``raw_data_version`` follows the ``v{number}`` format.

        Args:
            self: Candidate interim config instance.

        Returns:
            The validated interim config instance.
        """
        if not self.raw_data_version.startswith("v") or not self.raw_data_version[1:].isdigit():
            msg = (
                f"Invalid raw_data_version '{self.raw_data_version}'. "
                "It must start with 'v' followed by a number (e.g., v1, v2)."
            )
            logger.error(msg)
            raise ConfigError(msg)
        return self
