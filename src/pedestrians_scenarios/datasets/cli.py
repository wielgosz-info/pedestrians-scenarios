import pedestrians_scenarios.karma as km
from .basic_crossing import BasicSinglePedestrianCrossing


def datasets_generate_command(**kwargs):
    """Command line interface for generating datasets.

    :param kwargs: parsed command line arguments
    :type kwargs: Dict
    """
    generator = BasicSinglePedestrianCrossing(**kwargs)
    generator.generate()


def add_datasets_cli_args(command_subparsers, add_common_subcommand_args):
    """Prepares command line parameters for various datasets-related tasks.

    :type command_subparsers: argparse.SubParserAction
    """

    parser = command_subparsers.add_parser("datasets")
    subparsers = parser.add_subparsers()

    # 'generate' subcommand
    parser_generate = subparsers.add_parser("generate")
    parser_generate = add_common_subcommand_args(parser_generate)
    parser_generate = km.Karma.add_cli_args(parser_generate)
    parser_generate = BasicSinglePedestrianCrossing.add_cli_args(parser_generate)
    parser_generate.set_defaults(subcommand=datasets_generate_command)