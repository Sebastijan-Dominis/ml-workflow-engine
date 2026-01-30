from pydantic import BaseModel


class Broad(BaseModel):
    iterations: int
    n_iter: int


class Narrow(BaseModel):
    iterations: int
    n_iter: int


class SearchSchemaV1(BaseModel):
    broad: Broad
    narrow: Narrow
    cv: int
    scoring: str
    random_state: int