"""Validated output contract for technical entity extraction."""

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Criticality(str, Enum):
    BAJA = "baja"
    MEDIA = "media"
    ALTA = "alta"


class TechnicalEntities(BaseModel):
    """Technical facts extracted from one input paragraph."""

    model_config = ConfigDict(extra="forbid")

    tecnologias: list[NonEmptyText] = Field(
        description="Technologies explicitly named in the source text. Empty if none are named."
    )
    nivel_de_criticidad: Criticality = Field(
        description="Operational impact: baja, media, or alta."
    )
    resumen_tecnico: NonEmptyText = Field(
        description="Concise technical summary grounded in the source text."
    )
