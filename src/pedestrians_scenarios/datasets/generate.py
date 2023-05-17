try:
    import pedestrians_scenarios.karma as km
    from .generators.generator import Generator
    from .generators.binary_single_pedestrian import BinarySinglePedestrian
    from .generators.five_scenarios_single_pedestrian import FiveScenariosSinglePedestrian

    def add_cli_args(parser):
        """Add command line arguments for generating datasets.

        :param parser: command line parser
        :type parser: argparse.ArgumentParser
        :return: command line parser
        :rtype: argparse.ArgumentParser
        """

        parser.add_argument('--generator', default='binary_single_pedestrian', type=str,
                            choices=['binary_single_pedestrian',
                                     'five_scenarios_single_pedestrian'],
                            help='Generator to use (default: binary_single_pedestrian).')

        parser = km.karma.Karma.add_cli_args(parser)
        parser = Generator.add_cli_args(parser)
        # TODO: additional params can depend on the requested type of dataset in the future

        return parser

    def command(**kwargs):
        """Command line interface for generating datasets.

        :param kwargs: parsed command line arguments
        :type kwargs: Dict
        """

        # TODO: this can depend on the requested type of dataset in the future
        generator = {
            'binary_single_pedestrian': BinarySinglePedestrian,
            'five_scenarios_single_pedestrian': FiveScenariosSinglePedestrian
        }[kwargs.pop('generator')](**kwargs)

        generator.generate()

except ModuleNotFoundError as e:
    def add_cli_args(parser):
        return parser

    def command(**kwargs):
        raise NotImplementedError(
            "This command is only available when `carla` package is installed.") from e
