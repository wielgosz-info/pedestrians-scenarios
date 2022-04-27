import pedestrians_scenarios.karma as km
from .generators.basic_single_pedestrian_crossing import BasicSinglePedestrianCrossing

def add_cli_args(parser):
    parser = km.karma.Karma.add_cli_args(parser)

    # TODO: this can depend on the requested type of dataset in the future
    parser = BasicSinglePedestrianCrossing.add_cli_args(parser)

    return parser

def command(**kwargs):
    """Command line interface for generating datasets.

    :param kwargs: parsed command line arguments
    :type kwargs: Dict
    """

    # TODO: this can depend on the requested type of dataset in the future
    generator = BasicSinglePedestrianCrossing(**kwargs)

    generator.generate()