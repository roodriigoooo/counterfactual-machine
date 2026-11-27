"""Verified baseline evidence is decision context, never a predictor input."""
import hashlib
import json
import pymunk
from .schemas import Record
from .geometry import TOLERANCE


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def configuration_hash():
    from .simulator import PHYSICS_HZ, OBSERVATION_HZ
    return digest(dict(pymunk=pymunk.version, physics_hz=PHYSICS_HZ,
                       observation_hz=OBSERVATION_HZ, iterations=20, slop=.001,
                       monitor='closed-center physics polyline', tolerance=TOLERANCE))


class BaselineOutcome(Record):
    scene_hash: str
    goal_hash: str
    configuration_hash: str
    success: bool
    margin: float

    @classmethod
    def from_verification(cls, scene, goal, result):
        if not result['valid'] or result['margin'] is None:
            raise ValueError('valid simulator outcome required')
        return cls(scene_hash=scene.content_hash(), goal_hash=digest(goal.model_dump()),
                   configuration_hash=configuration_hash(), success=result['success'], margin=result['margin'])

    def validate_for(self, scene, goal):
        if (self.scene_hash, self.goal_hash, self.configuration_hash) != (
                scene.content_hash(), digest(goal.model_dump()), configuration_hash()):
            raise ValueError('baseline evidence does not match scene, goal, and simulator configuration')
