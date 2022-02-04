"""
Provide a deepcopy functions for the basic carla objects.
"""
import carla


def deepcopy_location(v: carla.Location) -> carla.Location:
    return carla.Location(
        x=v.x,
        y=v.y,
        z=v.z
    )


def deepcopy_rotation(v: carla.Rotation) -> carla.Rotation:
    return carla.Rotation(
        pitch=v.pitch,
        yaw=v.yaw,
        roll=v.roll,
    )


def deepcopy_transform(v: carla.Transform) -> carla.Transform:
    return carla.Transform(
        location=deepcopy_location(v.location),
        rotation=deepcopy_rotation(v.rotation)
    )
