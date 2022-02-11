import logging
import os
import time
from collections import namedtuple
from enum import Enum
from typing import Dict, Iterable, Tuple
from uuid import uuid4

import carla
import numpy as np
import pandas as pd
import pedestrians_scenarios.karma as km
from pedestrians_scenarios.karma.cameras.cameras_manager import (
    CamerasManager, FramesMergingMathod)
from pedestrians_scenarios.karma.karma import KarmaStage
from pedestrians_scenarios.karma.karma_data_provider import KarmaDataProvider
from srunner.scenariomanager.actorcontrols.pedestrian_control import \
    PedestrianControl
from tqdm.auto import trange

StandardDistribution = namedtuple('StandardDistribution', ['mean', 'std'])
PedestrianProfile = namedtuple(
    'PedestrianProfile', [
        'age', 'gender',
        'walking_speed', 'crossing_speed'
    ])

# Create some default profiles; those are up for revision
# somewhat based on what's found in doi:10.1016/j.sbspro.2013.11.160


class ExamplePedestrianProfiles(Enum):
    adult_female = PedestrianProfile('adult', 'female', StandardDistribution(
        1.19, 0.19), StandardDistribution(1.45, 0.23))
    adult_male = PedestrianProfile('adult', 'male', StandardDistribution(
        1.27, 0.21), StandardDistribution(1.47, 0.24))
    child_female = PedestrianProfile('child', 'female', StandardDistribution(
        0.9, 0.19), StandardDistribution(0.9, 0.23))
    child_male = PedestrianProfile('child', 'male', StandardDistribution(
        0.9, 0.21), StandardDistribution(0.9, 0.24))


class Generator(object):
    """
    Common class for various datasets generation. Handles managing/saving etc.
    """

    def __init__(self,
                 outputs_dir: str = './datasets',
                 number_of_clips: int = 512,
                 clip_length_in_frames: int = 600,
                 pedestrian_distribution: Iterable[Tuple[PedestrianProfile, float]] = (
                     (ExamplePedestrianProfiles.adult_female.value, 0.25),
                     (ExamplePedestrianProfiles.adult_male.value, 0.25),
                     (ExamplePedestrianProfiles.child_female.value, 0.25),
                     (ExamplePedestrianProfiles.child_male.value, 0.25)
                 ),
                 camera_position_distributions: Iterable[Iterable[StandardDistribution]] = ((
                     StandardDistribution(-10.0, 1.0),
                     StandardDistribution(0.0, 0.25),
                     StandardDistribution(1.0, 0.25)
                 ),),
                 batch_size: int = 16,
                 camera_fov: float = 90.0,
                 camera_image_size: Tuple[int, int] = (800, 600),
                 **kwargs
                 ) -> None:
        self._outputs_dir = outputs_dir
        os.makedirs(self._outputs_dir, exist_ok=True)

        self._number_of_clips = number_of_clips
        self._clip_length_in_frames = clip_length_in_frames
        self._pedestrian_distribution = pedestrian_distribution
        self._camera_distances_distributions = camera_position_distributions
        self._rng = KarmaDataProvider.get_rng()
        self._batch_size = batch_size
        self._camera_fov = camera_fov
        self._camera_image_size = camera_image_size

        self._total_batches = np.ceil(
            self._number_of_clips/self._batch_size).astype(int)
        self._karma = None
        self._kwargs = kwargs

    @staticmethod
    def add_cli_args(parser):
        subparser = parser.add_argument_group("Generator")

        subparser.add_argument('--outputs-dir', default=None, type=str,
                               help='Directory to store outputs (default: datasets)')
        subparser.add_argument('--number-of-clips', type=int, default=512,
                               help='Total number of clips to generate.')
        subparser.add_argument('--clip-length-in-frames', type=int, default=600,
                               help='Length of each clip in frames.')
        subparser.add_argument('--batch-size', type=int, default=16,
                               help='Number of clips in each batch.')

        return parser

    def get_profiles(self, batch_idx: int) -> Iterable[Iterable[PedestrianProfile]]:
        """
        Get the pedestrian profiles for each clip.
        By default, this will return a random, single pedestrian profile per clip.
        """
        profiles, weights = zip(*self._pedestrian_distribution)
        indices = self._rng.choice(np.arange(len(profiles)), size=(
            self._batch_size,), replace=True, p=weights)
        return [(profiles[i],) for i in indices]

    def get_spawn_points(self, batch_idx: int, profiles: Iterable[Iterable[PedestrianProfile]]) -> Iterable[Iterable[carla.Transform]]:
        """
        Get the spawn points for each pedestrian.
        By default, spawn points will be randomly chosen from list of allowed spawn points.

        :param profiles: List of pedestrian profiles that will be used.
        :type profiles: Iterable[Iterable[PedestrianProfile]]
        """
        spawn_points = []
        for clip_profiles in profiles:
            in_clip = []
            for _ in clip_profiles:
                in_clip.append(KarmaDataProvider.get_pedestrian_spawn_point())
            spawn_points.append(in_clip)
        return spawn_points

    def get_models(self, batch_idx: int, profiles: Iterable[Iterable[PedestrianProfile]]) -> Iterable[Iterable[str]]:
        """
        Get the models (blueprint names) for each pedestrian.

        :param profiles: List of pedestrian profiles that will be used.
        :type profiles: Iterable[Iterable[PedestrianProfile]]
        """
        return [
            [
                km.Walker.get_model_by_age_and_gender(profile.age, profile.gender)
                for profile in clip_profiles
            ] for clip_profiles in profiles
        ]

    def get_pedestrians(self, batch_idx: int, models: Iterable[Iterable[str]], spawn_points: Iterable[Iterable[carla.Transform]]) -> Iterable[Iterable[km.Walker]]:
        """
        Generate pedestrians for a batch.
        In general, this should not tick the world, but rather just create the pedestrians.
        Karma.tick() will be called after this to actually spawn the pedestrians.

        :param batch_idx: [description]
        :type batch_idx: int
        :param models: [description]
        :type models: Iterable[Iterable[str]]
        :param spawn_points: [description]
        :type spawn_points: Iterable[Iterable[carla.Transform]]
        :return: [description]
        :rtype: Iterable[Iterable[km.Walker]]
        """
        pedestrians = []
        for clip_idx, (clip_models, clip_spawn_points) in enumerate(zip(models, spawn_points)):
            in_clip = []
            try:
                for model, spawn_point in zip(clip_models, clip_spawn_points):
                    in_clip.append(
                        km.Walker(model=model, spawn_point=spawn_point, tick=False))
            except RuntimeError:
                logging.getLogger(__name__).warning(
                    f'Failed to create pedestrians for clip {clip_idx} in batch {batch_idx}')
            pedestrians.append(in_clip)
        return pedestrians

    def get_camera_distances(self, batch_idx: int) -> Iterable[Iterable[float]]:
        """
        Get the camera distances for each clip.
        By default, this will return a random, single camera position per clip.
        """
        distances = []
        total_cameras = len(self._camera_distances_distributions)
        for camera_idx in range(total_cameras):
            camera_distances = []
            for i in range(3):  # x,y,z
                camera_distances.append(self._rng.normal(
                    loc=self._camera_distances_distributions[camera_idx][i].mean,
                    scale=self._camera_distances_distributions[camera_idx][i].std,
                    size=(self._batch_size,)
                ))
            distances.append(list(zip(*camera_distances)))
        return list(zip(*distances))

    def get_camera_look_at(self, batch_idx: int, pedestrians: Iterable[Iterable[km.Walker]], camera_distances: Iterable[Iterable[float]]) -> Iterable[Iterable[carla.Transform]]:
        """
        Get the camera look at points for each clip.
        By default, this will return a random, single camera position per clip.
        """
        camera_look_at = []
        for clip_idx, (clip_pedestrians, clip_camera_distances) in enumerate(zip(pedestrians, camera_distances)):
            camera_look_at.append(self.get_clip_camera_look_at(
                batch_idx, clip_idx, clip_pedestrians, clip_camera_distances))
        return camera_look_at

    def setup_pedestrians(self, batch_idx: int, pedestrians: Iterable[Iterable[km.Walker]], profiles: Iterable[Iterable[PedestrianProfile]], camera_look_at: Iterable[Iterable[carla.Transform]]) -> None:
        """
        Setup the pedestrians for a batch. It is called after the pedestrians are spawned in the world.
        """
        for clip_idx, (clip_pedestrians, clip_profiles, clip_look_at) in enumerate(zip(pedestrians, profiles, camera_look_at)):
            self.setup_clip_pedestrians(
                batch_idx, clip_idx, clip_pedestrians, clip_profiles, clip_look_at)

    def get_pedestrians_control(self, batch_idx: int, pedestrians: Iterable[Iterable[km.Walker]], profiles: Iterable[Iterable[PedestrianProfile]]) -> Iterable[Iterable[PedestrianControl]]:
        """
        Sets up controllers for all pedestrians.
        :param batch_idx: [description]
        :type batch_idx: int
        :param pedestrians: [description]
        :type pedestrians: Iterable[Iterable[km.Walker]]
        :param profiles: [description]
        :type profiles: Iterable[Iterable[PedestrianProfile]]
        :return: [description]
        :rtype: Iterable[Iterable[PedestrianControl]]
        """
        return [
            self.get_clip_pedestrians_control(
                batch_idx, clip_idx, clip_pedestrians, clip_profiles
            )
            for clip_idx, (clip_pedestrians, clip_profiles) in enumerate(zip(pedestrians, profiles))
        ]

    def get_clip_pedestrians_control(self, batch_idx: int, clip_idx: int, pedestrians: Iterable[km.Walker], profiles: Iterable[PedestrianProfile]) -> Iterable[PedestrianControl]:
        """
        Get the pedestrians controls for a single clip.
        """
        raise NotImplementedError()

    def setup_clip_pedestrians(self, batch_idx: int, clip_idx: int, pedestrians: Iterable[km.Walker], profiles: Iterable[PedestrianProfile], camera_look_at: Iterable[carla.Transform]) -> None:
        """
        Setup the pedestrians in a single clip.
        """
        raise NotImplementedError()

    def get_clip_camera_look_at(self, batch_idx: int, clip_idx: int, pedestrians: Iterable[km.Walker], camera_distances: Iterable[float]) -> Iterable[carla.Transform]:
        """
        Get the camera look at points for a single clip.
        """
        raise NotImplementedError()

    def get_camera_managers(
        self,
        batch_idx: int,
        camera_look_at: Iterable[Iterable[carla.Transform]],
        camera_distances: Iterable[Iterable[float]],
        pedestrians: Iterable[Iterable[km.Walker]]
    ) -> Iterable[Iterable[CamerasManager]]:
        """
        Get camera managers for batch. By default this will create a single
        CameraManager per clip, merging images from all specified clip cameras.
        This method ticks the world with each CameraManager created,
        since cameras have to be spawned to attach event listeners.

        :param batch_idx: [description]
        :type batch_idx: int
        :param camera_look_at: [description]
        :type camera_look_at: Iterable[Iterable[carla.Transform]]
        :param camera_distances: [description]
        :type camera_distances: Iterable[Iterable[float]]
        """
        managers = []
        for clip_idx, (clip_look_at, clip_distances, clip_pedestrians) in enumerate(zip(camera_look_at, camera_distances, pedestrians)):
            managers.append(self.get_clip_camera_managers(
                batch_idx, clip_idx, clip_look_at, clip_distances, clip_pedestrians
            ))
        return managers

    def get_clip_camera_managers(self, batch_idx: int, clip_idx: int, camera_look_at: Iterable[carla.Transform], camera_distances: Iterable[float], pedestrians: Iterable[km.Walker]) -> Iterable[CamerasManager]:
        """
        Get the camera managers for a single clip. By default all created cameras are FreeCamera.
        This method ticks the world since cameras have to be spawned to attach event listeners.
        """
        if len(camera_look_at):
            manager = CamerasManager(
                self._karma,
                outputs_dir=os.path.join(self._outputs_dir, 'clips'),
                merging_method=FramesMergingMathod.none
            )
            manager.create_free_cameras(
                cameras=zip(camera_look_at, camera_distances),
                image_size=self._camera_image_size,
                fov=self._camera_fov,
            )
            return [manager, ]
        else:
            return []

    def get_map_for_batch(self, batch_idx: int) -> str:
        """
        Get the map for a batch. All pedestrians in batch will be spawned in the same world at the same time.
        """
        return self._rng.choice(KarmaDataProvider.get_available_maps())

    def generate(self) -> None:
        """
        Generate the dataset.
        """

        outfile = os.path.join(self._outputs_dir, 'data.csv')

        with km.Karma(**self._kwargs) as karma:
            self._karma = karma
            for batch_idx in trange(self._total_batches, desc="Batch"):
                map_name = self.get_map_for_batch(batch_idx)
                karma.reset_world(map_name)

                batch_data = self.generate_batch(batch_idx, map_name)
                pd.DataFrame(batch_data).to_csv(outfile, mode='a',
                                                header=(batch_idx == 0),
                                                index=False)

        self._karma = None

    def generate_batch(self, batch_idx: int, map_name: str) -> Iterable[Dict]:
        """
        Generate a single batch.
        """
        profiles = self.get_profiles(batch_idx)
        spawn_points = self.get_spawn_points(batch_idx, profiles)
        models = self.get_models(batch_idx, profiles)
        pedestrians = self.get_pedestrians(batch_idx, models, spawn_points)

        # spawn pedestrians in world
        self._karma.tick()

        camera_distances = self.get_camera_distances(batch_idx)
        camera_look_at = self.get_camera_look_at(
            batch_idx, pedestrians, camera_distances)
        camera_managers = self.get_camera_managers(
            batch_idx, camera_look_at, camera_distances, pedestrians)
        # no need to tick, CamerasManager already did it

        self.setup_pedestrians(batch_idx, pedestrians, profiles, camera_look_at)

        # apply settings to pedestrians
        self._karma.tick()

        controllers = self.get_pedestrians_control(batch_idx, pedestrians, profiles)
        for clip_controllers in controllers:
            for controller in clip_controllers:
                self._karma.register_controller(controller)

        # sleep for a little bit to apply everything before recording starts
        time.sleep(2)

        # start cameras
        recordings = []
        for clip_idx, managers in enumerate(camera_managers):
            clip_recordings = []
            for manager_idx, manager in enumerate(managers):
                recording_name = f'{batch_idx}-{clip_idx}-{manager_idx}'
                clip_recordings.append(recording_name)
                manager.start_recording(recording_name)
            recordings.append(clip_recordings)

        # prepare data capture callback
        frame_data = [[] for _ in range(self._batch_size)]
        capture_callback_id = self._karma.register_callback(
            KarmaStage.tick, lambda snapshot: self.capture_frame_data(snapshot, pedestrians, frame_data))

        # move simulation forward the required number of frames
        # and capture per-frame data
        for _ in trange(self._clip_length_in_frames, desc="Frame"):
            self._karma.tick()

        # sleep for a little bit to let cameras & data capture finish recording
        time.sleep(2)

        # stop data capture
        self._karma.unregister_callback(capture_callback_id)

        # stop cameras
        recorded_frames = []
        for managers in camera_managers:
            clip_recorded_frames = []
            for manager in managers:
                clip_recorded_frames.append(manager.stop_recording())
            recorded_frames.append(clip_recorded_frames)

        # unregister controllers
        for clip_controllers in controllers:
            for controller in clip_controllers:
                self._karma.unregister_controller(controller)

        # collect batch data
        batch_data = self.collect_batch_data(
            map_name, profiles, spawn_points, models, pedestrians, camera_distances, camera_look_at, recordings, recorded_frames, frame_data)

        return batch_data

    def capture_frame_data(self, snapshot: carla.WorldSnapshot, pedestrians: Iterable[Iterable[km.Walker]], out_frame_data: Iterable[Iterable[Iterable[Dict]]]):
        """
        Capture per-frame data.
        """
        for clip_pedestrians, clip_data in zip(pedestrians, out_frame_data):
            current_frame_data = []
            for pedestrian in clip_pedestrians:
                pedestrian_snapshot = snapshot.find(pedestrian.id)
                pedestrian_transform = pedestrian_snapshot.get_transform()
                pedestrian_velocity = pedestrian_snapshot.get_velocity()

                current_frame_data.append({
                    'world.frame': snapshot.frame,
                    'frame.pedestrian.id': pedestrian.id,
                    'frame.pedestrian.location.x': pedestrian_transform.location.x,
                    'frame.pedestrian.location.y': pedestrian_transform.location.y,
                    'frame.pedestrian.location.z': pedestrian_transform.location.z,
                    'frame.pedestrian.rotation.pitch': pedestrian_transform.rotation.pitch,
                    'frame.pedestrian.rotation.yaw': pedestrian_transform.rotation.yaw,
                    'frame.pedestrian.rotation.roll': pedestrian_transform.rotation.roll,
                    'frame.pedestrian.velocity.x': pedestrian_velocity.x,
                    'frame.pedestrian.velocity.y': pedestrian_velocity.y,
                    'frame.pedestrian.velocity.z': pedestrian_velocity.z,
                })
                # TODO: add the actual data of interest for each frame - 3D pose, 2D pose

            clip_data.append(current_frame_data)

    def collect_batch_data(self, map_name, profiles, spawn_points, models, pedestrians, camera_distances, camera_look_at, recordings, recorded_frames, captured_data):
        batch_data = []
        for clip_idx in range(self._batch_size):
            # recordings base name; at the moment we assume a single CameraManager per clip,
            # and therefore a single recording basename;
            # optionally there can be no recordings (e.g due to spawning/timeout errors)
            # TODO: handle multiple managers per clip
            manager_idx = 0
            recording: str = recordings[clip_idx][manager_idx] if len(
                recordings[clip_idx]) else None

            if recording is None:
                logging.getLogger(__name__).warning(
                    f'No recording was created fot clip {clip_idx}, skipping.')
                continue

            clip_recorded_frames: Iterable[int] = sorted(
                recorded_frames[clip_idx][manager_idx])

            clip_models: Iterable[str] = models[clip_idx]
            clip_profiles: Iterable[PedestrianProfile] = profiles[clip_idx]
            clip_spawn_points: Iterable[carla.Transform] = spawn_points[clip_idx]
            clip_pedestrians: Iterable[km.Walker] = pedestrians[clip_idx]
            clip_camera_distances: Iterable[Tuple[float]] = camera_distances[clip_idx]
            clip_camera_look_at: Iterable[carla.Transform] = camera_look_at[clip_idx]

            clip_captured_data: Iterable[Dict] = sorted(
                captured_data[clip_idx], key=lambda x: x[0]['world.frame'])
            clip_captured_data_frames: Iterable[int] = [
                data[0]['world.frame']
                for data in clip_captured_data
            ]

            clip_id = str(uuid4())  # make it unique in dataset

            # TODO: in multi-manager setup, the number of cameras in each manager
            # is not necessarily the same as len(clip_camera_distances) == len(clip_camera_look_at).
            # Also, there can be cameras other than FreeCamera, that also produce recordings.
            # Also, some of the cameras can move (e.g. attached to a vehicle), so camera position should be per frame also?
            for camera_idx, (distance, look_at) in enumerate(zip(clip_camera_distances, clip_camera_look_at)):
                min_frame = clip_recorded_frames[0]
                max_frame = clip_recorded_frames[-1]

                for frame_idx, world_frame in enumerate(range(min_frame, max_frame + 1)):
                    # extract frame data or minimal dict if not available
                    if world_frame in clip_captured_data_frames:
                        frame = clip_captured_data[clip_captured_data_frames.index(
                            world_frame)]
                    else:
                        frame = [{
                            'frame.pedestrian.id': p.id,
                            'world.frame': world_frame
                        } for p in range(len(clip_pedestrians))]

                    for pedestrian_idx, (profile, model, spawn_point, pedestrian) in enumerate(zip(clip_profiles, clip_models, clip_spawn_points, clip_pedestrians)):
                        assert frame[pedestrian_idx]['frame.pedestrian.id'] == pedestrian.id, "Pedestrian ID mismatch"
                        assert frame[pedestrian_idx]['world.frame'] == world_frame, "World frame mismatch"

                        full_frame_data = {
                            'id': clip_id,
                            'world.map': map_name,
                            'camera.idx': camera_idx,
                            'camera.recording': f'{recording}-{camera_idx}.mp4',
                            'camera.distance.x': distance[0],
                            'camera.distance.y': distance[1],
                            'camera.distance.z': distance[2],
                            'camera.look_at.x': look_at.location.x,
                            'camera.look_at.y': look_at.location.y,
                            'camera.look_at.z': look_at.location.z,
                            'camera.look_at.pitch': look_at.rotation.pitch,
                            'camera.look_at.yaw': look_at.rotation.yaw,
                            'camera.look_at.roll': look_at.rotation.roll,
                            'pedestrian.idx': pedestrian_idx,
                            'pedestrian.model': model,
                            'pedestrian.age': profile.age,
                            'pedestrian.gender': profile.gender,
                            'pedestrian.spawn_point.x': spawn_point.location.x,
                            'pedestrian.spawn_point.y': spawn_point.location.y,
                            'pedestrian.spawn_point.z': spawn_point.location.z,
                            'pedestrian.spawn_point.pitch': spawn_point.rotation.pitch,
                            'pedestrian.spawn_point.yaw': spawn_point.rotation.yaw,
                            'pedestrian.spawn_point.roll': spawn_point.rotation.roll,
                            'frame.idx': frame_idx,
                        }
                        full_frame_data.update(frame[pedestrian_idx])

                        batch_data.append(full_frame_data)
        return batch_data
