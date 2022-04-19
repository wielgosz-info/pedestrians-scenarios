from typing import Iterable, List

import numpy as np

from pedestrians_scenarios.karma.karma_data_provider import KarmaDataProvider
from pedestrians_scenarios.karma.utils.deepcopy import deepcopy_transform
from pedestrians_scenarios.pedestrian_controls.basic_pedestrian_control import BasicPedestrianControl
from pedestrians_scenarios.datasets.generator import Generator, BatchGenerator, PedestrianProfile
from pedestrians_scenarios.karma.walker import Walker
import carla


class BasicSinglePedestrianCrossingBatch(BatchGenerator):
    """
    Creates dataset with randomized pedestrians crossing the street.
    Pedestrians are controlled by BasicPedestrianControl.
    Each clip should contain a single pedestrian crossing the street,
    HOWEVER, due to concurrent data generation, the pedestrians other
    than primary one may be visible in the clip.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def get_clip_camera_look_at(self, clip_idx: int, pedestrians: Iterable[Walker], camera_distances: Iterable[float]) -> Iterable[carla.Transform]:
        """
        Get the camera look at points for a single clip.
        In BasicSinglePedestrianCrossing there can be only one pedestrian in the clip,
        but potentially there can be more than one camera.
        All cameras are looking at the same point.
        """
        waypoints: List[carla.Waypoint] = [
            KarmaDataProvider.get_shifted_driving_lane_waypoint(
                pedestrian.get_transform().location,
                waypoint_jitter_scale=self._waypoint_jitter_scale)
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

    def setup_clip_pedestrians(self, clip_idx: int, pedestrians: Iterable[Walker], profiles: Iterable[PedestrianProfile], camera_look_at: Iterable[carla.Transform]) -> None:
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
            # compute how we need to rotate the pedestrian to face the waypoint
            pedestrian_transform = deepcopy_transform(pedestrian.get_transform())
            delta = np.rad2deg(np.arctan2(direction_unit.y, direction_unit.x))
            pedestrian_transform.rotation.yaw = pedestrian_transform.rotation.yaw + delta
            pedestrian.set_transform(pedestrian_transform)

    def get_clip_pedestrians_control(self, clip_idx: int, pedestrians: Iterable[Walker], profiles: Iterable[PedestrianProfile], camera_look_at: Iterable[carla.Transform]) -> Iterable[BasicPedestrianControl]:
        """
        Get the pedestrians controls for a single clip.
        """
        waypoint = camera_look_at[0]

        controllers = []
        for pedestrian, profile in zip(pedestrians, profiles):
            controller = BasicPedestrianControl(pedestrian)
            controller.update_target_speed(self._rng.normal(
                profile.crossing_speed.mean, profile.crossing_speed.std))
            
            pathInfo = self.generatePath(pedestrian, waypoint)
            controller.update_waypoints(pathInfo[0])
            controller.setLaneWaypoint(pathInfo[1])

            controllers.append(controller)

        return controllers


    def Distance(self, pos1, pos2):

        pos1_np = np.asarray([pos1.x, pos1.y, pos1.z])
        pos2_np = np.asarray([pos2.x, pos2.y, pos2.z])

        dist = np.power(pos2_np - pos1_np, 2)

        dist = np.sum(dist)

        dist = np.sqrt(dist)

        return dist

    def getRoadParallelPath(self, pedestrian, waypoint):

        try:

            roadpos1 = waypoint
            roadpos2 = KarmaDataProvider.get_closest_driving_lane_waypoint(waypoint.location).next(2)[0].transform

            vectorRoad = roadpos2.location - roadpos1.location

            pedestrianLocation = pedestrian.get_transform().location

            distPos1 = self.Distance(roadpos1.location, pedestrianLocation)
            distPos2 = self.Distance(roadpos2.location, pedestrianLocation)

            if distPos1 > distPos2:

                dir = vectorRoad * -1

            else:

                dir = vectorRoad

            parallelWaypoint = carla.Transform(location=(pedestrianLocation + dir * 5 * KarmaDataProvider.get_rng().randn()), rotation=carla.Rotation())

            return [parallelWaypoint, roadpos2]


        except IndexError:

            print('IndexError in getRoadParallelPath')

            return [pedestrian.clip_spawn_points[0], waypoint]
    

    def generatePath(self, pedestrian, waypoint):

        pr = KarmaDataProvider.get_rng().randn()

        if pr < 0.2:
            # Case 0: Pedestrian directly wants to cross the street:
            path = [waypoint]
            laneWaypointPos = 0

        elif pr >= 0.2 and pr < 0.25:
            # Case 1: Pedestrian starts crossing the street and then regrets and goes back again:
            path = [waypoint, pedestrian.spawn_point]
            laneWaypointPos = 0

        elif pr >= 0.25 and pr < 0.75:
            # Case 2: Pedestrian walks to a point in the path and then decides crossing the street:

            pedNextPos, roadpos2 = self.getRoadParallelPath(pedestrian, waypoint)
            nextWaypoint = roadpos2 if KarmaDataProvider.get_rng().randn() < 0.75 else waypoint

            path = [pedNextPos, nextWaypoint]
            laneWaypointPos = 1

        elif pr >= 0.75 and pr < 0.8:
            # Case 3: Pedestrian walks to a point in the path, then decides crossing the street, and finally regrets and goes back:

            pedNextPos, roadpos2 = self.getRoadParallelPath(pedestrian, waypoint)
            nextWaypoint = roadpos2 if KarmaDataProvider.get_rng().randn() < 0.75 else waypoint

            path = [pedNextPos, nextWaypoint, pedNextPos]
            laneWaypointPos = 1

        elif pr >= 0.8:
            # Case 4: Pedestrian walks to a point in the path and never decides to cross the street:

            pedNextPos, roadpos2 = self.getRoadParallelPath(pedestrian, waypoint)
            nextWaypoint = roadpos2 if KarmaDataProvider.get_rng().randn() < 0.75 else waypoint
            
            path = [pedNextPos]
            laneWaypointPos = -1

        return [path, laneWaypointPos]
    


class BasicSinglePedestrianCrossing(Generator):
    batch_generator = BasicSinglePedestrianCrossingBatch
