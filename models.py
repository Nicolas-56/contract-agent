"""Shared contract schema for server data and client-side result validation."""

from datetime import date

from pydantic import BaseModel, Field


class Contract(BaseModel):
    """A synthetic contract stored in the local JSON dataset."""

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    parties: list[str] = Field(min_length=2)
    signed_on: date
    text: str = Field(min_length=1)
