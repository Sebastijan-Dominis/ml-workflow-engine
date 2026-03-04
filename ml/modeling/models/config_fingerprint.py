from pydantic import BaseModel


class ConfigFingerprint(BaseModel):
    config_hash: str
    pipeline_cfg_hash: str = ""
