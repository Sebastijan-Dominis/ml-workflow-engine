from typing import Literal

from pydantic import BaseModel


class RunIdentity(BaseModel):
    """Model representing the identity of a run."""
    train_run_id: str
    snapshot_id: str
    status: Literal["success"]
