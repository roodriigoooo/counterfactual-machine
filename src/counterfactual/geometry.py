import numpy as np

from .schemas import EditSpec, SceneSpec

TOLERANCE = 1e-9


def segment_distance(p, a, b):
    p, a, b = np.asarray(p), np.asarray(a), np.asarray(b)
    u = np.clip(np.sum((p-a)*(b-a), axis=-1)/np.sum((b-a)**2), 0, 1)
    return np.linalg.norm(p-a-u[..., None]*(b-a), axis=-1)


def validate_scene(scene: SceneSpec):
    ball, bumper = scene.ball, scene.bumper
    for p, r in [(ball.position, ball.radius), (bumper.position, bumper.radius)]:
        if any(v-r < 0 or v+r > size for v, size in zip(p, scene.world)):
            raise ValueError("initial circles must fit inside the authoring world")
    if np.linalg.norm(np.subtract(ball.position, bumper.position)) < ball.radius+bumper.radius-TOLERANCE:
        raise ValueError("initial ball/bumper overlap")
    for s in scene.segments:
        if any(v < 0 or v > size for p in [s.a, s.b] for v, size in zip(p, scene.world)):
            raise ValueError("segment endpoints must lie in the authoring world")
        for obj in [ball, bumper]:
            if segment_distance(obj.position, s.a, s.b) < obj.radius+s.radius-TOLERANCE:
                raise ValueError(f"initial {obj.id}/{s.id} overlap")
    return scene


def apply_edit(scene: SceneSpec, edit: EditSpec) -> SceneSpec:
    if edit.object_id != scene.bumper.id:
        raise ValueError("only the designated bumper may be edited")
    if abs(edit.displacement) > scene.ball.radius+TOLERANCE:
        raise ValueError("edit exceeds one ball radius")
    x, y = scene.bumper.position
    bumper = scene.bumper.model_copy(update={"position": (x+edit.displacement, y)})
    return validate_scene(scene.model_copy(update={"bumper": bumper}))


def candidates(scene: SceneSpec, count: int = 257):
    if count < 3 or count % 2 != 1:
        raise ValueError("grid must be odd and include zero and both bounds")
    grid = np.linspace(-scene.ball.radius, scene.ball.radius, count)
    valid, reasons = [], []
    for u in grid:
        try:
            apply_edit(scene, EditSpec(object_id=scene.bumper.id, displacement=float(u)))
            valid.append(True)
            reasons.append(None)
        except ValueError as e:
            valid.append(False)
            reasons.append(str(e))
    return grid, np.array(valid), reasons
