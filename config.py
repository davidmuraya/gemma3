from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    EXCHANGE_RATE_SITE: str = Field(..., env="EXCHANGE_RATE_SITE")

    class Config:
        env_file = ".env"


@lru_cache
def get_settings():
    return Settings()
