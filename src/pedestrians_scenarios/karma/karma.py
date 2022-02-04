import argparse
from time import time

import carla

from .karma_data_provider import KarmaDataProvider


class Karma(object):
    """
    Primary client/world/everything manager.

    Heavily inspired by main ScenarioRunner class & various gym Envs for CARLA.
    Assumption: this is a single-server single-client configuration, since we want to
    take the advantage of the ScenarioRunner's KarmaDataProvider helper,
    and it stores them as class variables.

    The sync mode if forced to be True, since TrafficManager is only supposed to be used
    in synchronous mode and conditionally using sync/not sync is very error prone.
    In the future, the async mode may be supported (possibly as explicit separate class),
    but for now, we'll keep it simple.
    """

    def __init__(
        self,
        host='server',
        port=2000,
        timeout=10.0,
        traffic_manager_port=8000,
        seed=22752,
        fps=30.0,
        hybrid_physics_mode=False,
        **kwargs
    ) -> None:
        self.__timeout = timeout
        self.__seed = seed
        self.__fps = fps
        self.__hybrid_physics_mode = hybrid_physics_mode

        self.__on_tick_callback_id = None

        self.__client = carla.Client(host, port)
        self.__client.set_timeout(self.__timeout)

        self.__traffic_manager = self.__client.get_trafficmanager(traffic_manager_port)
        KarmaDataProvider.set_traffic_manager_port(traffic_manager_port)

        self.__world = None
        self.reset_world()

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        Adds Karma-specific command line arguments.
        """
        subparser = parser.add_argument_group("Karma")

        subparser.add_argument('--host', default='server',
                               help='Hostname or IP of the CARLA server (default: server)')
        subparser.add_argument('--port', default=2000, type=int,
                               help='TCP port to listen to (default: 2000)')
        subparser.add_argument('--timeout', default=10.0, type=float,
                               help='Set the CARLA client timeout value in seconds')
        subparser.add_argument('--traffic-manager-port', default=8000, type=int,
                               help='Port to use for the TrafficManager (default: 8000)')
        subparser.add_argument('--seed', default=22752, type=int,
                               help='Seed used by the everything (default: 22752)')
        subparser.add_argument('--fps', default=30.0, type=float,
                               help='FPS of the simulation (default: 30)')
        subparser.add_argument('--hybrid-physics-mode', default=False, action='store_true',
                               help='Enable hybrid physics mode (default: False)')

        return parser

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

        return False

    def tick(self) -> int:
        return self.__world.tick()

    def reset_world(self, map_name=None):
        prev_id = None

        if self.__world is not None:
            prev_id = self.__world.id
            self.__world.remove_on_tick(self.__on_tick_callback_id)

            if map_name is not None and KarmaDataProvider.get_map().name != map_name:
                self.__client.load_world(map_name)
            else:
                self.__client.reload_world()

        self.__world = self.__client.get_world()
        tries = self.__timeout
        while prev_id == self.__world.id and tries > 0:
            tries -= 1
            time.sleep(1)
            self.__world = self.__client.get_world()
        if tries < 1:
            raise RuntimeError("Could not reset world.")

        # always reset settings
        settings = carla.WorldSettings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / self.__fps
        settings.deterministic_ragdolls = True
        self.__world.apply_settings(settings)

        self.__world.set_pedestrians_seed(self.__seed)

        KarmaDataProvider.cleanup()
        KarmaDataProvider.set_client(self.__client)
        KarmaDataProvider.set_world(self.__world)

        self.__traffic_manager.set_synchronous_mode(True)
        self.__traffic_manager.set_random_device_seed(self.__seed)
        self.__traffic_manager.set_hybrid_physics_mode(self.__hybrid_physics_mode)

        self.__on_tick_callback_id = self.__world.on_tick(self.on_carla_tick)

    def close(self):
        KarmaDataProvider.cleanup()

        if self.__world:
            self.__world.remove_on_tick(self.__on_tick_callback_id)
            self.__world.apply_settings(carla.WorldSettings(
                synchronous_mode=False,
                fixed_delta_seconds=0.0
            ))
        if self.__traffic_manager:
            self.__traffic_manager.set_synchronous_mode(False)

    def on_carla_tick(self, snapshot: carla.WorldSnapshot):
        KarmaDataProvider.on_carla_tick(snapshot)
