# Modularize if new datas are added in the future, or if the interim config becomes too large. This will help keep the code organized and maintainable.

import logging
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, model_validator

from ml.data.utils.config.schemas.constants import BorderValue
from ml.data.utils.config.schemas.shared import DataInfo
from ml.exceptions import ConfigError
from ml.registry.interim_constraints import (ALLOWED_VALUES_CONSTRAINTS,
                                             MAX_CONSTRAINTS, MIN_CONSTRAINTS)

logger = logging.getLogger(__name__)

class DataSchema(BaseModel):
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
    lowercase_columns: bool = Field(True, description="Whether to convert column names to lowercase.")
    strip_strings: bool = Field(True, description="Whether to strip leading/trailing whitespace from string columns.")
    replace_spaces_in_columns: bool = Field(True, description="Whether to replace spaces in column names with underscores.")
    replace_dashes_in_columns: bool = Field(True, description="Whether to replace dashes in column names with underscores.")



class Invariant(BaseModel):
    min: Optional[BorderValue] = Field(None, description="Minimum allowed value for the column.")
    max: Optional[BorderValue] = Field(None, description="Maximum allowed value for the column.")
    allowed_values: Optional[list] = Field(None, description="List of allowed values for the column.")

class Invariants(BaseModel):
    hotel: Optional[Invariant] = None
    is_canceled: Optional[Invariant] = None
    lead_time: Optional[Invariant] = None
    arrival_date_year: Optional[Invariant] = None
    arrival_date_month: Optional[Invariant] = None
    arrival_date_week_number: Optional[Invariant] = None
    arrival_date_day_of_month: Optional[Invariant] = None
    stays_in_weekend_nights: Optional[Invariant] = None
    stays_in_week_nights: Optional[Invariant] = None
    adults: Optional[Invariant] = None
    children: Optional[Invariant] = None
    babies: Optional[Invariant] = None
    meal: Optional[Invariant] = None
    country: Optional[Invariant] = None
    market_segment: Optional[Invariant] = None
    distribution_channel: Optional[Invariant] = None
    is_repeated_guest: Optional[Invariant] = None
    previous_cancellations: Optional[Invariant] = None
    previous_bookings_not_canceled: Optional[Invariant] = None
    reserved_room_type: Optional[Invariant] = None
    assigned_room_type: Optional[Invariant] = None
    booking_changes: Optional[Invariant] = None
    deposit_type: Optional[Invariant] = None
    agent: Optional[Invariant] = None
    company: Optional[Invariant] = None
    days_in_waiting_list: Optional[Invariant] = None
    customer_type: Optional[Invariant] = None
    adr: Optional[Invariant] = None
    required_car_parking_spaces: Optional[Invariant] = None
    total_of_special_requests: Optional[Invariant] = None
    reservation_status: Optional[Invariant] = None
    reservation_status_date: Optional[Invariant] = None
    name: Optional[Invariant] = None
    email: Optional[Invariant] = None
    phone_number: Optional[Invariant] = None
    credit_card: Optional[Invariant] = None

    @model_validator(mode="after")
    # validate that the invariants specified in the config do not violate the predefined constraints
    def validate_constraints(self):
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
        for field_name in DataSchema.model_fields.keys():
            if field_name not in values:
                values[field_name] = Invariant(
                    min=MIN_CONSTRAINTS.get(field_name),
                    max=MAX_CONSTRAINTS.get(field_name),
                    allowed_values=ALLOWED_VALUES_CONSTRAINTS.get(field_name)
                )
        return values

class LineageConfig(BaseModel):
    created_by: str
    created_at: datetime

class InterimConfig(BaseModel):
    data: DataInfo
    data_schema: DataSchema
    cleaning: Cleaning
    invariants: Invariants
    drop_duplicates: bool = Field(True, description="Whether to drop duplicate rows from the data (default: True).")
    drop_missing_ints: bool = Field(True, description="Whether to drop rows with missing values in integer columns (default: True).")
    min_rows: int = Field(0, description="Minimum number of rows required after cleaning (default: 0).")
    lineage: LineageConfig