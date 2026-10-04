"""CARLA connection, world settings and actor lifecycle management.

Importing this module makes CARLA's bundled ``agents`` package importable
(needed for ``GlobalRoutePlanner``).
"""
from __future__ import annotations

import glob
import logging
import math
import os
import sys
from typing import Any, Iterable, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)

_AGENT_PATH_CANDIDATES = [
    os.environ.get("CARLA_ROOT", ""),
    os.path.join(os.environ.get("CARLA_ROOT", ""), "PythonAPI", "carla"),
    os.path.expanduser("~/Documents/CARLA_0.9.16/PythonAPI/carla"),
    "/opt/carla-simulator/PythonAPI/carla",
    "/opt/carla/PythonAPI/carla",
]


def ensure_carla_agents_on_path() -> str:
    """Make CARLA's ``agents`` package importable and return the path used."""
    try:
        import agents.navigation.global_route_planner  # noqa: F401
        return os.path.dirname(sys.modules["agents"].__file__)
    except ModuleNotFoundError:
        pass

    candidates = list(_AGENT_PATH_CANDIDATES)
    candidates += glob.glob(os.path.expanduser("~/*/CARLA_*/PythonAPI/carla"))
    for path in candidates:
        if path and os.path.isdir(os.path.join(path, "agents")):
            if path not in sys.path:
                sys.path.append(path)
            try:
                import agents.navigation.global_route_planner  # noqa: F401
                return path
            except ModuleNotFoundError:
                continue
    raise RuntimeError(
        "Could not locate CARLA's 'agents' package. Set CARLA_ROOT to your "
        "CARLA installation directory (e.g. ~/Documents/CARLA_0.9.16)."
    )


ensure_carla_agents_on_path()

import carla  # noqa: E402
from agents.navigation.global_route_planner import GlobalRoutePlanner  # noqa: E402,F401
from agents.navigation.local_planner import RoadOption  # noqa: E402,F401


class CarlaSession:
    """Owns the client/world handle and restores the original settings.

    A single session is shared by the environment for its whole lifetime;
    ``restore()`` puts the simulator back into asynchronous mode so that a
    crashed or interrupted training run does not leave the server wedged.
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.client = carla.Client(cfg.host, cfg.port)
        self.client.set_timeout(cfg.timeout)

        if not self._map_matches(cfg.town):
            if not cfg.allow_load_world:
                current = self.client.get_world().get_map().name.split("/")[-1]
                raise RuntimeError(
                    f"CARLA is running {current!r} but the config asks for "
                    f"{cfg.town!r}.\nRelaunch the server with the right map:\n"
                    f"    $CARLA_ROOT/CarlaUE4.sh {cfg.town} -quality-level=Low\n"
                    "Switching maps from the client takes minutes and can crash "
                    "the server on a machine with limited RAM.  Set "
                    "carla.allow_load_world=true to try anyway, or "
                    f"--town {current} to use the loaded map."
                )
            LOGGER.info("Loading world %s (this can take several minutes)", cfg.town)
            self.world = self.client.load_world(cfg.town)
        elif cfg.reload_world:
            LOGGER.info("Reloading world")
            self.world = self.client.reload_world()
        else:
            self.world = self.client.get_world()

        self._original_settings = self.world.get_settings()

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = cfg.fixed_delta_seconds
        settings.no_rendering_mode = cfg.no_rendering
        # Keeps sensor substepping stable at 20 Hz.
        settings.substepping = True
        settings.max_substep_delta_time = 0.01
        settings.max_substeps = 10
        self.world.apply_settings(settings)

        self.map = self.world.get_map()
        self.blueprints = self.world.get_blueprint_library()
        self.spawn_points = self.map.get_spawn_points()
        self.spectator = self.world.get_spectator()
        self._restored = False

    def _map_matches(self, town: str) -> bool:
        try:
            current = self.client.get_world().get_map().name
        except RuntimeError:
            return False
        return current.split("/")[-1] == town.split("/")[-1]

    def tick(self) -> int:
        return self.world.tick()

    def restore(self) -> None:
        if self._restored:
            return
        try:
            self.world.apply_settings(self._original_settings)
        except Exception as exc:  # pragma: no cover - server may be gone
            LOGGER.warning("Could not restore world settings: %s", exc)
        self._restored = True


class ActorRegistry:
    """Tracks spawned actors so an episode can always be torn down fully."""

    def __init__(self, world) -> None:
        self.world = world
        self._sensors: List[Any] = []
        self._actors: List[Any] = []

    def add(self, actor, is_sensor: bool = False):
        if actor is None:
            return None
        (self._sensors if is_sensor else self._actors).append(actor)
        return actor

    def destroy_all(self) -> None:
        # Sensors must stop listening before being destroyed, otherwise
        # callbacks can fire against freed memory.
        for sensor in self._sensors:
            try:
                if sensor.is_listening:
                    sensor.stop()
            except Exception:
                pass
        for actor in self._sensors + self._actors:
            try:
                actor.destroy()
            except Exception:
                pass
        self._sensors.clear()
        self._actors.clear()


# --------------------------------------------------------------------------- #
#  Small conversions between CARLA types and numpy
# --------------------------------------------------------------------------- #
def location_xy(location) -> np.ndarray:
    return np.array([location.x, location.y], dtype=np.float64)


def actor_xy(actor) -> np.ndarray:
    return location_xy(actor.get_transform().location)


def actor_yaw_deg(actor) -> float:
    return float(actor.get_transform().rotation.yaw)


def actor_speed_ms(actor) -> float:
    v = actor.get_velocity()
    return float(math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z))


def actor_velocity_xy(actor) -> np.ndarray:
    v = actor.get_velocity()
    return np.array([v.x, v.y], dtype=np.float64)


def route_to_polyline(route: Iterable[Tuple[Any, Any]]) -> np.ndarray:
    """Convert a ``GlobalRoutePlanner`` route into an (N, 2) array."""
    return np.array([[wp.transform.location.x, wp.transform.location.y]
                     for wp, _ in route], dtype=np.float64)


def has_line_of_sight(world, from_location, to_location,
                      z_offset: float = 1.0) -> bool:
    """True if no static geometry blocks the straight line between two points.

    Uses CARLA's ray cast against level geometry.  Other vehicles are
    intentionally ignored: they are dynamic and only marginally attenuate
    ITS-G5 / C-V2X signals compared with buildings.
    """
    start = carla.Location(from_location.x, from_location.y,
                           from_location.z + z_offset)
    end = carla.Location(to_location.x, to_location.y,
                         to_location.z + z_offset)
    if start.distance(end) < 0.5:
        return True
    try:
        hits = world.cast_ray(start, end)
    except AttributeError:  # pragma: no cover - very old CARLA
        return True
    for hit in hits:
        label = hit.label
        if label in (carla.CityObjectLabel.Buildings,
                     carla.CityObjectLabel.Walls,
                     carla.CityObjectLabel.Fences,
                     carla.CityObjectLabel.Truck,
                     carla.CityObjectLabel.Bus,
                     carla.CityObjectLabel.Static):
            return False
    return True


def set_spectator_behind(spectator, actor, distance: float = 8.0,
                         height: float = 4.0, pitch: float = -20.0) -> None:
    transform = actor.get_transform()
    yaw_rad = math.radians(transform.rotation.yaw)
    location = carla.Location(
        x=transform.location.x - distance * math.cos(yaw_rad),
        y=transform.location.y - distance * math.sin(yaw_rad),
        z=transform.location.z + height,
    )
    spectator.set_transform(
        carla.Transform(location,
                        carla.Rotation(pitch=pitch, yaw=transform.rotation.yaw)))
