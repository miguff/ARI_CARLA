"""Collision detection."""
from __future__ import annotations

from typing import Optional

from ..carla_utils import carla


class CollisionSensor:
    def __init__(self, world, attach_to) -> None:
        blueprint = world.get_blueprint_library().find("sensor.other.collision")
        self.actor = world.spawn_actor(blueprint, carla.Transform(),
                                       attach_to=attach_to)
        self.happened = False
        self.other_actor_type: Optional[str] = None
        self.impulse = 0.0
        self.actor.listen(self._on_collision)

    def _on_collision(self, event) -> None:
        self.happened = True
        self.other_actor_type = event.other_actor.type_id
        impulse = event.normal_impulse
        self.impulse = float((impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2) ** 0.5)

    @property
    def hit_cyclist(self) -> bool:
        if not self.other_actor_type:
            return False
        return ("bike" in self.other_actor_type
                or "bicycle" in self.other_actor_type
                or "crossbike" in self.other_actor_type)

    def destroy(self) -> None:
        try:
            if self.actor.is_listening:
                self.actor.stop()
            self.actor.destroy()
        except Exception:
            pass
