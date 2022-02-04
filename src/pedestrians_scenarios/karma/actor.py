import carla
from typing import Generic, TypeVar, Any

from .karma_data_provider import KarmaDataProvider

A = TypeVar('A', carla.Actor, carla.Walker, carla.Vehicle)


class Actor(Generic[A]):
    """
    Proxy for carla.Actor objects that goes through the
    KarmaDataProvider to access common properities.
    """

    def __init__(self, actor: A = None, **kwargs) -> None:
        if actor is None:
            actor = KarmaDataProvider.request_new_actor(**kwargs)
        self.__actor = actor

    def get_transform(self) -> carla.Transform:
        return KarmaDataProvider.get_transform(self.__actor)

    def get_location(self) -> carla.Location:
        return KarmaDataProvider.get_location(self.__actor)

    def get_velocity(self) -> float:
        return KarmaDataProvider.get_velocity(self.__actor)

    @property
    def id(self) -> int:
        return self.__actor.id

    def __getattr__(self, __name: str) -> Any:
        return getattr(self.__actor, __name)
