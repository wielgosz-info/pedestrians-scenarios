import carla
from typing import Generic, TypeVar, Any

from pedestrians_scenarios.third_party.srunner.scenariomanager.carla_data_provider import CarlaDataProvider

A = TypeVar('A', carla.Actor, carla.Walker, carla.Vehicle)


class Actor(Generic[A]):
    """
    Proxy for carla.Actor objects that goes through the
    CarlaDataProvider to access common properities.
    """

    def __init__(self, actor: A = None, **kwargs) -> None:
        if actor is None:
            actor = CarlaDataProvider.request_new_actor(**kwargs)
        self.__actor = actor

    def get_transform(self) -> carla.Transform:
        return CarlaDataProvider.get_transform(self.__actor)

    def get_location(self) -> carla.Location:
        return CarlaDataProvider.get_location(self.__actor)

    def get_velocity(self) -> float:
        return CarlaDataProvider.get_velocity(self.__actor)

    def __getattr__(self, __name: str) -> Any:
        return getattr(self.__actor, __name)
