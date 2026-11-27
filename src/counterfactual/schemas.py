"""Versioned, immutable contracts. Geometry is in metres; time is in seconds."""
from typing import Literal
import hashlib

from pydantic import BaseModel, ConfigDict, Field, model_validator

Vec2 = tuple[float, float]


class Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class Material(Record):
    friction: float = Field(default=0.4, ge=0, le=1)
    restitution: float = Field(default=0.7, ge=0, le=1)


class Ball(Record):
    id: str = "ball"
    position: Vec2 = (1.7, 3.2)
    velocity: Vec2 = (2.4, 0.0)
    radius: float = Field(default=0.12, ge=0.08, le=0.2)
    mass: float = Field(default=1.0, gt=0)
    angle: float = 0.0
    angular_velocity: float = 0.0
    material: Material = Material()


class Bumper(Record):
    id: str = "bumper"
    position: Vec2 = (2.7, 2.15)
    radius: float = Field(default=0.32, ge=0.15, le=0.5)
    material: Material = Material()


class Segment(Record):
    id: str
    a: Vec2
    b: Vec2
    radius: float = Field(default=0.04, ge=0.02, le=0.1)
    material: Material = Material()

    @model_validator(mode="after")
    def nondegenerate(self):
        if sum((a-b)**2 for a, b in zip(self.a, self.b)) < 0.01:
            raise ValueError("segment must be at least 0.1 m long")
        return self


class SceneSpec(Record):
    schema_version: Literal[1] = 1
    id: str
    parent_id: str
    family: Literal["approach", "ramp", "authored"] = "approach"
    seed: int = 0
    ball: Ball = Ball()
    bumper: Bumper = Bumper()
    segments: tuple[Segment, ...] = Field(default=(), max_length=4)
    gravity: Vec2 = (0.0, -9.81)
    horizon: Literal[2.0] = 2.0
    world: Vec2 = (6.0, 4.0)

    @model_validator(mode="after")
    def supported(self):
        ids = [self.ball.id, self.bumper.id, *(s.id for s in self.segments)]
        if len(set(ids)) != len(ids):
            raise ValueError("object IDs must be unique")
        if self.gravity != (0.0, -9.81) or self.world != (6.0, 4.0):
            raise ValueError("baseline fixes gravity and the 6 × 4 m authoring world")
        if max(abs(v) for v in self.ball.velocity) > 5:
            raise ValueError("baseline launch speed components are limited to 5 m/s")
        return self

    def content_hash(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()


class EditSpec(Record):
    object_id: str = "bumper"
    property: Literal["position.x"] = "position.x"
    displacement: float = 0.0


class Region(Record):
    kind: Literal["disk", "halfplane"] = "disk"
    center: Vec2 = (4.0, 0.8)
    radius: float = Field(default=0.3, gt=0)
    normal: Vec2 = (1.0, 0.0)
    offset: float = 4.0

    @model_validator(mode="after")
    def unit_normal(self):
        if self.kind == "halfplane" and abs(sum(v*v for v in self.normal)-1) > 1e-8:
            raise ValueError("half-plane normal must be unit length; inside is n·p >= offset")
        return self


class GoalSpec(Record):
    ball_id: str = "ball"
    kind: Literal["reach", "avoid"] = "reach"
    region: Region = Region()
    start: float = Field(default=0.0, ge=0, le=2)
    end: float = Field(default=2.0, ge=0, le=2)
    boundary: Literal["closed_center"] = "closed_center"

    @model_validator(mode="after")
    def interval(self):
        if self.start > self.end:
            raise ValueError("goal interval must be ordered")
        return self
