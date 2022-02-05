import argparse
import logging
import sys
import time
from typing import List

from pedestrians_scenarios import __version__
import pedestrians_scenarios.karma as km
from pedestrians_scenarios.karma.utils.deepcopy import deepcopy_transform
from pedestrians_scenarios.karma.cameras import CamerasManager, FramesMergingMathod
from pedestrians_scenarios.karma.karma_data_provider import KarmaDataProvider
import carla
from tqdm.auto import trange
import numpy as np
from pedestrians_scenarios.pedestrian_controls import BasicPedestrianControl

_logger = logging.getLogger(__name__)


# ---- CLI ----
# The functions defined in this section are wrappers around the main Python
# API allowing them to be called directly from the terminal as a CLI
# executable/script.


def add_cli_args():
    """Prepares command line parameters.

    Returns:
      :parser:`argparse.ArgumentParser`
    """
    parser = argparse.ArgumentParser(description="Just a Fibonacci demonstration")
    parser.add_argument(
        "--version",
        action="version",
        version="pedestrians-scenarios {ver}".format(ver=__version__),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO,
    )
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG,
    )
    return parser


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=loglevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )


def main(args):
    parser = add_cli_args()
    km.Karma.add_cli_args(parser)

    args = parser.parse_args(args)
    kwargs = vars(args)
    setup_logging(args.loglevel)

    with km.Karma(**kwargs) as karma:
        models = [
            km.Walker.get_model_by_age_and_gender(a, g)
            for (a, g) in (('adult', 'female'), ('adult', 'male'), ('child', 'female'), ('child', 'male'))
        ]

        pedestrians: List[km.Walker] = []
        for model in models:
            pedestrians.append(km.Walker(
                model=model, spawn_point=None,
                random_location=True, tick=False
            ))

        karma.tick()

        camera_managers: List[CamerasManager] = []
        waypoints: List[carla.Waypoint] = []

        for pedestrian in pedestrians:
            waypoint = KarmaDataProvider.get_shifted_driving_lane_waypoint(
                pedestrian.get_transform().location)

            waypoints.append(waypoint)

            manager = CamerasManager(
                karma, merging_method=FramesMergingMathod.horizontal)
            manager.create_free_cameras((
                (waypoint.transform, [-10.0, -10.0, 10]),
                (waypoint.transform, [-10.0, 10.0, 10]),
                (waypoint.transform, [-10.0, 0.0, 1.2])
            ))  # this ticks the world
            camera_managers.append(manager)

        for pedestrian, waypoint in zip(pedestrians, waypoints):
            direction_unit = (waypoint.transform.location -
                              pedestrian.get_transform().location)
            direction_unit.z = 0  # ignore height
            direction_unit = direction_unit.make_unit_vector()

            # shortcut, since we're ignoring elevation
            pedestrian_transform = deepcopy_transform(pedestrian.get_transform())
            delta = np.rad2deg(np.arctan2(direction_unit.y, direction_unit.x))
            pedestrian_transform.rotation.yaw = pedestrian_transform.rotation.yaw + delta
            pedestrian.set_transform(pedestrian_transform)

        # tick here, to ensure all pedestrians have rotated
        karma.tick()

        # create contollers for each pedestrian and register them with Karma
        for pedestrian in pedestrians:
            controller = BasicPedestrianControl(pedestrian)
            controller.update_target_speed(1.2)
            karma.register_controller(controller)

        # start cameras
        for (pedestrian, manager) in zip(pedestrians, camera_managers):
            manager.start_recording(pedestrian.id)

        # move simulation forward
        for idx in trange(0, 60):
            karma.tick()

        time.sleep(1)


def run():
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`

    This function can be used as entry point to create console scripts with setuptools.
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    # ^  This is a guard statement that will prevent the following code from
    #    being executed in the case someone imports this file instead of
    #    executing it as a script.
    #    https://docs.python.org/3/library/__main__.html

    # After installing your project with pip, users can also run your Python
    # modules as scripts via the ``-m`` flag, as defined in PEP 338::
    #
    #     python -m pedestrians_scenarios.skeleton
    #
    run()
