from typing import List
import carla


def convert_transform_to_flat_list(transform: carla.Transform) -> List[float]:
    return [
        transform.location.x,
        transform.location.y,
        transform.location.z,
        transform.rotation.pitch,
        transform.rotation.yaw,
        transform.rotation.roll,
    ]


def convert_flat_list_to_transform(transform_flat: List[float]) -> carla.Transform:
    return carla.Transform(
        carla.Location(x=transform_flat[0], y=transform_flat[1], z=transform_flat[2]),
        carla.Rotation(pitch=transform_flat[3],
                       yaw=transform_flat[4], roll=transform_flat[5])
    )
