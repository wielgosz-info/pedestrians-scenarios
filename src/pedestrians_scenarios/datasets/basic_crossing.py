from typing import Iterable, List

import numpy as np

from pedestrians_scenarios.karma.karma_data_provider import KarmaDataProvider
from pedestrians_scenarios.karma.utils.deepcopy import deepcopy_transform
from pedestrians_scenarios.pedestrian_controls.basic_pedestrian_control import BasicPedestrianControl
from pedestrians_scenarios.datasets.generator import Generator, PedestrianProfile
import pedestrians_scenarios.karma as km
import carla


class BasicSinglePedestrianCrossing(Generator):
    """
    Creates dataset with randomized pedestrians crossing the street.
    Pedestrians are controlled by BasicPedestrianControl.
    Each clip should contain a single pedestrian crossing the street,
    HOWEVER, due to concurrent data generation, the pedestrians other
    than primary one may be visible in the clip.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def get_clip_camera_look_at(self, batch_idx: int, clip_idx: int, pedestrians: Iterable[km.Walker], camera_distances: Iterable[float]) -> Iterable[carla.Transform]:
        """
        Get the camera look at points for a single clip.
        In BasicSinglePedestrianCrossing there can be only one pedestrian in the clip,
        but potentially there can be more than one camera.
        All cameras are looking at the same point.
        """
        waypoints: List[carla.Waypoint] = [
            KarmaDataProvider.get_shifted_driving_lane_waypoint(
                pedestrian.get_transform().location)
            for pedestrian in pedestrians
        ]

        assert len(waypoints) < 2, "Only one pedestrian per clip is supported"

        if len(waypoints) == 0:
            return []

        waypoint = waypoints[0]
        camera_look_at: List[carla.Transform] = [
            waypoint.transform for _ in camera_distances
        ]

        return camera_look_at

    def setup_clip_pedestrians(self, batch_idx: int, clip_idx: int, pedestrians: Iterable[km.Walker], profiles: Iterable[PedestrianProfile], camera_look_at: Iterable[carla.Transform]) -> None:
        """
        Setup the pedestrians in a single clip.
        This method is called before get_clip_pedestrians_control().
        It should not tick the world.
        """
        waypoint = camera_look_at[0]

        for pedestrian in pedestrians:
            direction_unit = (waypoint.location -
                              pedestrian.get_transform().location)
            direction_unit.z = 0  # ignore height
            direction_unit = direction_unit.make_unit_vector()

            # shortcut, since we're ignoring elevation
            pedestrian_transform = deepcopy_transform(pedestrian.get_transform())
            delta = np.rad2deg(np.arctan2(direction_unit.y, direction_unit.x))
            pedestrian_transform.rotation.yaw = pedestrian_transform.rotation.yaw + delta
            pedestrian.set_transform(pedestrian_transform)

    def get_clip_pedestrians_control(self, batch_idx: int, clip_idx: int, pedestrians: Iterable[km.Walker], profiles: Iterable[PedestrianProfile], camera_look_at: Iterable[carla.Transform]) -> Iterable[BasicPedestrianControl]:
        """
        Get the pedestrians controls for a single clip.
        """
        waypoint = camera_look_at[0]

        controllers = []
        for pedestrian, profile in zip(pedestrians, profiles):
            controller = BasicPedestrianControl(pedestrian)
            controller.update_target_speed(self._rng.normal(
                profile.crossing_speed.mean, profile.crossing_speed.std))
            controller.update_waypoints([
                waypoint
            ])
            controllers.append(controller)
        return controllers
