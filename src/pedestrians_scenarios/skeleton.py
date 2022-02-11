import argparse
import logging
import sys
import time
from typing import List

from pedestrians_scenarios import __version__
from pedestrians_scenarios.datasets.basic_crossing import BasicSinglePedestrianCrossing
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
    parser = km.Karma.add_cli_args(parser)
    parser = BasicSinglePedestrianCrossing.add_cli_args(parser)

    args = parser.parse_args(args)
    kwargs = vars(args)
    setup_logging(args.loglevel)

    gen = BasicSinglePedestrianCrossing(**kwargs)
    gen.generate()


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
