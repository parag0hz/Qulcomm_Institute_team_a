"""Pydantic request contracts shared by Paragon API endpoints."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    StringConstraints,
    field_validator,
)


NonEmptyText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=2000),
]


class DesignParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    B_Ramp_Angle: FiniteFloat
    B_Diffusor_Angle: FiniteFloat
    B_Trunklid_Angle: FiniteFloat
    C_Side_Mirrors_Rotation: FiniteFloat
    D_Rear_Window_Inclination: FiniteFloat
    D_Winscreen_Inclination: FiniteFloat
    C_Side_Mirrors_Translate_X: FiniteFloat
    C_Side_Mirrors_Translate_Z: FiniteFloat
    D_Winscreen_Length: FiniteFloat
    D_Rear_Window_Length: FiniteFloat
    E_A_B_C_Pillar_Thickness: FiniteFloat
    G_Trunklid_Curvature: FiniteFloat
    G_Trunklid_Length: FiniteFloat
    H_Front_Bumper_Curvature: FiniteFloat
    H_Front_Bumper_Length: FiniteFloat
    F_Door_Handles_Thickness: FiniteFloat
    F_Door_Handles_Z_Position: FiniteFloat
    E_Fenders_Arch_Offset: FiniteFloat
    A_Car_Length: FiniteFloat
    F_Door_Handles_X_Position: FiniteFloat
    A_Car_Width: FiniteFloat
    A_Car_Roof_Height: FiniteFloat
    A_Car_Green_House_Angle: FiniteFloat
    CarRear: Literal["Fastback", "Estateback", "Notchback"]
    Wheels: Literal["Open detailed", "Open smooth", "Closed smooth"]


class OptimizeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parameters: DesignParameters
    target_cd: FiniteFloat = Field(ge=0.18, le=0.36)
    locked: list[str] = Field(default_factory=list, max_length=23)

    @field_validator("locked")
    @classmethod
    def unique_locks(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("Locked parameters must be unique.")
        allowed = set(DesignParameters.model_fields) - {"CarRear", "Wheels"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"Unknown locked parameters: {', '.join(unknown)}")
        return value


class CopilotHistoryItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: NonEmptyText


class CopilotRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: NonEmptyText
    parameters: DesignParameters
    history: list[CopilotHistoryItem] = Field(default_factory=list, max_length=6)


class VertexTestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parameters: DesignParameters | None = None
