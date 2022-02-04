import argparse
import logging
import sys
import os
import av

from pedestrians_scenarios import __version__
import pedestrians_scenarios.karma as km
from pedestrians_scenarios.karma.karma_data_provider import KarmaDataProvider
import numpy as np
import carla
from PIL import Image

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

        pedestrians = []
        for model in models:
            pedestrians.append(km.Walker(
                model=model, spawn_point=None,
                random_location=True, tick=False
            ))

        karma.tick()

        cameras = []
        waypoints = []
        directions = []
        for pedestrian in pedestrians:
            pedestrian_transform = pedestrian.get_transform()
            shifted_waypoint = KarmaDataProvider.get_shifted_driving_lane_waypoint(
                pedestrian_transform.location)

            waypoints.append(shifted_waypoint)

            direction_unit = (shifted_waypoint.transform.location -
                              pedestrian_transform.location).make_unit_vector()
            direction_unit.z = 0  # ignore height

            distance_to_travel = 150
            direction = direction_unit * distance_to_travel
            directions.append(direction)

            pedestrian.apply_control(carla.WalkerControl(
                direction=direction_unit,
                speed=2.1,
                jump=False
            ))

            cameras.append(km.FreeCamera(
                look_at=shifted_waypoint.transform,
                distance=[-10.0, -10.0, 10],
                tick=False
            ))
            cameras.append(km.FreeCamera(
                look_at=shifted_waypoint.transform,
                distance=[-10.0, 0.0, 5.7],
                tick=False
            ))

        # ensure cameras spawned & control is applied
        karma.tick()

        # ensure camera has transform
        karma.tick()

        os.makedirs('/outputs/scenarios/', exist_ok=True)

        vid = {}
        for pi, ped in enumerate(pedestrians):
            vid[ped.id] = []

        for idx in range(0, 300):
            frame = []
            for camera in cameras:
                d = camera.get_data()
                frame.append(d)
            n_frame = np.array(frame).astype(np.uint8)
            n_frame = n_frame.reshape(
                (4, 2, n_frame.shape[1], n_frame.shape[2], n_frame.shape[3]))
            n_frame = n_frame.transpose((1, 0, 2, 3, 4))
            n_frame = np.concatenate(n_frame, axis=2)

            for pi, ped in enumerate(pedestrians):
                vid[ped.id].append(n_frame[pi])

            karma.tick()

        for ped_id, frames in vid.items():
            frames = np.array(frames).astype(np.uint8)

            # copied from torchvision
            with av.open(os.path.join('/outputs/scenarios/', f'{str(ped_id)}.mp4'), mode="w") as container:
                stream = container.add_stream('libx264', rate=30)
                stream.width = frames.shape[2]
                stream.height = frames.shape[1]
                stream.pix_fmt = "yuv420p"
                stream.options = {}

                for img in frames:
                    frame = av.VideoFrame.from_ndarray(img, format="rgb24")
                    frame.pict_type = "NONE"
                    for packet in stream.encode(frame):
                        container.mux(packet)

                # Flush stream
                for packet in stream.encode():
                    container.mux(packet)


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
