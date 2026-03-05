"""Shared schema constants and typed value wrappers for data configs."""

from typing import Literal

from pydantic import BaseModel


class BorderValue(BaseModel):
    """Boundary value and comparison operator used in invariant checks."""

    value: float
    op: Literal["lt", "lte", "gte", "ge"]
