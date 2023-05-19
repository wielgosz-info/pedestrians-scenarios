import logging
import sys


def setup_logging(loglevel=logging.WARNING, logger: logging.Logger = logging.getLogger()):
    """Setup basic logging
    :param loglevel: Minimum loglevel for emitting messages.
    :type loglevel: int
    """
    logger.setLevel(loglevel if loglevel else logging.WARNING)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        fmt="[%(asctime)s] %(levelname)s:%(name)s:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))

    logger.addHandler(handler)
