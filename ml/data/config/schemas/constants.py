from typing import Literal

from pydantic import BaseModel


class BorderValue(BaseModel):
    value: float
    op: Literal["lt", "lte", "gte", "ge"]