"""Fresh Pymunk space per branch; canonical IDs; callbacks at physics resolution."""
from dataclasses import dataclass, field
import numpy as np
import pymunk

from .geometry import apply_edit, validate_scene
from .schemas import EditSpec, SceneSpec

PHYSICS_HZ = 1920
OBSERVATION_HZ = 60


@dataclass
class Trajectory:
    times: np.ndarray
    states: np.ndarray  # x,y,vx,vy,angle,angular velocity
    contacts: np.ndarray  # interval × canonical static object
    object_ids: list[str]
    events: list[dict]
    physics_times: np.ndarray
    physics_positions: np.ndarray
    provenance: dict
    responses: list[dict] = field(default_factory=list)

    def public(self):
        return {"kind": "simulation", "times": self.times.tolist(), "states": self.states.tolist(),
                "contacts": self.contacts.tolist(), "object_ids": self.object_ids,
                "events": self.events, "provenance": self.provenance}


def simulate(scene: SceneSpec, displacement: float = 0.0, physics_hz: int = PHYSICS_HZ, record_responses: bool = False):
    if physics_hz < 60 or physics_hz % 60:
        raise ValueError("physics frequency must be a positive multiple of 60 Hz")
    validate_scene(scene)
    edited = apply_edit(scene, EditSpec(object_id=scene.bumper.id, displacement=displacement))
    space = pymunk.Space()
    space.gravity = scene.gravity
    space.iterations = 20
    space.collision_slop = 0.001
    ball = scene.ball
    body = pymunk.Body(ball.mass, pymunk.moment_for_circle(ball.mass, 0, ball.radius))
    body.position, body.velocity = ball.position, ball.velocity
    body.angle, body.angular_velocity = ball.angle, ball.angular_velocity
    shape = pymunk.Circle(body, ball.radius)
    shape.friction, shape.elasticity = ball.material.friction, ball.material.restitution
    shape.collision_type = 1
    space.add(body, shape)
    objects = sorted([edited.bumper, *edited.segments], key=lambda o: o.id)
    ids = [o.id for o in objects]
    lookup = {}
    for o in objects:
        if o.id == edited.bumper.id:
            static_body = pymunk.Body(body_type=pymunk.Body.STATIC)
            static_body.position = o.position
            s = pymunk.Circle(static_body, o.radius)
            space.add(static_body)
        else:
            s = pymunk.Segment(space.static_body, o.a, o.b, o.radius)
        s.friction, s.elasticity = o.material.friction, o.material.restitution
        s.collision_type = 2
        space.add(s)
        lookup[s] = o.id
    n = round(scene.horizon*physics_hz)
    ratio = physics_hz//60
    contacts = np.zeros((120, len(ids)), dtype=np.float32)
    events = []
    responses = []
    before = None
    step = 0

    def identify(arb):
        return next(lookup[s] for s in arb.shapes if s in lookup)

    def mark(arb, _space, _data):
        obj = identify(arb)
        contacts[min((step-1)//ratio, 119), ids.index(obj)] = 1

    def begin(arb, _space, _data):
        events.append({"time": step/physics_hz, "phase": "begin", "objects": [ball.id, identify(arb)]})

    def separate(arb, _space, _data):
        events.append({"time": step/physics_hz, "phase": "separate", "objects": [ball.id, identify(arb)]})

    def response(arb, _space, _data):
        responses.append({"time": step/physics_hz, "object": identify(arb),
                          "normal_ball_to_static": list(arb.normal),
                          "impulse_on_ball": list(arb.total_impulse),
                          "first_contact": bool(arb.is_first_contact),
                          "before": before, "after": state()})

    space.on_collision(1, 2, begin=begin, pre_solve=mark, separate=separate,
                       post_solve=response if record_responses else None)
    states = np.zeros((121, 6))
    positions = np.zeros((n+1, 2))

    def state():
        return [*body.position, *body.velocity, body.angle, body.angular_velocity]

    states[0], positions[0] = state(), body.position
    for step in range(1, n+1):
        if record_responses:
            before = state()
        space.step(1/physics_hz)
        positions[step] = body.position
        if step % ratio == 0:
            states[step//ratio] = state()
    # Copy records before destruction (space teardown may call separation callbacks).
    return Trajectory(np.linspace(0, 2, 121), states, contacts, ids, list(events),
                      np.linspace(0, 2, n+1), positions,
                      {"scene_hash": scene.content_hash(), "edit_m": displacement,
                       "pymunk": pymunk.version, "physics_hz": physics_hz,
                       "observation_hz": 60, "iterations": 20, "collision_slop_m": 0.001,
                       "monitor": "closed-center piecewise-linear, physics rate", "schema_version": 1}, list(responses))
