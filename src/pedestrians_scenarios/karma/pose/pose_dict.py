from typing import Dict, List, Tuple, Union

import carla
from pedestrians_scenarios import karma as km

from .skeleton import CARLA_SKELETON

PoseDict = Dict[CARLA_SKELETON, carla.Transform]
Pose2DDict = Dict[CARLA_SKELETON, carla.Vector2D]


def get_pedestrian_pose_dicts(pedestrian: Union[km.Walker, carla.Walker]) -> Tuple[PoseDict, PoseDict, PoseDict]:
    # TODO: should this be here or cached as a part of KarmaDataProvider?
    bones = pedestrian.get_bones().bone_transforms

    world_pose = [None] * len(CARLA_SKELETON)
    component_pose = [None] * len(CARLA_SKELETON)
    relative_pose = [None] * len(CARLA_SKELETON)

    for bone in bones:
        # we only care about the body pose, so ignore other bones
        if bone.name not in CARLA_SKELETON.__members__:
            continue
        bone_idx = CARLA_SKELETON[bone.name].value
        world_pose[bone_idx] = bone.world
        component_pose[bone_idx] = bone.component
        relative_pose[bone_idx] = bone.relative

    return (
        dict(zip(CARLA_SKELETON, world_pose)),
        dict(zip(CARLA_SKELETON, component_pose)),
        dict(zip(CARLA_SKELETON, relative_pose)),
    )


def convert_pose_dict_to_flat_list(pose: PoseDict) -> List[float]:
    pose_flat = []

    for bone in pose.values():
        pose_flat.append(bone.location.x)
        pose_flat.append(bone.location.y)
        pose_flat.append(bone.location.z)
        pose_flat.append(bone.rotation.pitch)
        pose_flat.append(bone.rotation.yaw)
        pose_flat.append(bone.rotation.roll)

    return pose_flat


def convert_flat_list_to_pose_dict(pose_flat: List[float]) -> PoseDict:
    pose = {}

    for bone in CARLA_SKELETON:
        i = bone.value * 6
        pose[bone] = carla.Transform(
            carla.Location(x=pose_flat[i + 0], y=pose_flat[i + 1], z=pose_flat[i + 2]),
            carla.Rotation(pitch=pose_flat[i + 3],
                           yaw=pose_flat[i + 4], roll=pose_flat[i + 5])
        )

    return pose


def convert_pose_2d_dict_to_flat_list(pose: Pose2DDict) -> List[float]:
    pose_flat = []

    for bone in pose.values():
        pose_flat.append(bone.x)
        pose_flat.append(bone.y)

    return pose_flat


def convert_flat_list_to_pose_2d_dict(pose_flat: List[float]) -> Pose2DDict:
    pose = {}

    for bone in CARLA_SKELETON:
        i = bone.value * 2
        pose[bone] = carla.Vector2D(x=pose_flat[i + 0], y=pose_flat[i + 1])

    return pose
