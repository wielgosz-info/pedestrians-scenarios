import pedestrians_scenarios.karma as km
from .basic_crossing import BasicSinglePedestrianCrossing


def datasets_generate_command(**kwargs):
    """Command line interface for generating datasets.

    Args:
      args (:obj:`argparse.Namespace`): parsed command line arguments
    """
    generator = BasicSinglePedestrianCrossing(**kwargs)
    generator.generate()


def add_datasets_cli_args(command_subparsers):
    """Prepares command line parameters for various datasets-related tasks.

    Returns:
      :subparsers:`argparse.SubParserAction`
    """

    parser = command_subparsers.add_parser("datasets")
    subparsers = parser.add_subparsers()

    # 'generate' subcommand
    parser_generate = subparsers.add_parser("generate")
    parser_generate = km.Karma.add_cli_args(parser_generate)
    parser_generate = BasicSinglePedestrianCrossing.add_cli_args(parser_generate)
    parser_generate.set_defaults(subcommand=datasets_generate_command)
