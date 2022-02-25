import ast
import logging
import os
import time
from collections import namedtuple
from enum import Enum
from typing import Dict, Iterable, List, Tuple
from uuid import uuid4
import multiprocessing as mp

import carla
import numpy as np
import pandas as pd
import pedestrians_scenarios.karma as km
from pedestrians_scenarios.karma.cameras import (
    CamerasManager, FramesMergingMathod)
from pedestrians_scenarios.karma.pose.pose_dict import convert_list_to_pose_dict, convert_pose_2d_dict_to_list, convert_pose_dict_to_list, get_pedestrian_pose_dicts
from srunner.scenariomanager.actorcontrols.pedestrian_control import \
    PedestrianControl
from tqdm.auto import tqdm, trange
from pedestrians_scenarios.karma.pose.projection import project_pose

from pedestrians_scenarios.karma.utils.conversions import convert_transform_to_list, convert_vector3d_to_list

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


class BatchGenerator(mp.Process):
    """
    Common class that generates single batch of clips/data.
    """

    def __init__(self,
                 outfile: str,
                 outfile_lock: mp.Lock,
                 outputs_dir: str,
                 queue: mp.Queue,
                 seed: int = 22752,
                 batch_idx: int = 0,
                 batch_size: int = 16,
                 clip_length_in_frames: int = 600,
                 pedestrian_distributions: Iterable[Tuple[PedestrianProfile, float]] = (
                     (ExamplePedestrianProfiles.adult_female.value, 0.25),
                     (ExamplePedestrianProfiles.adult_male.value, 0.25),
                     (ExamplePedestrianProfiles.child_female.value, 0.25),
                     (ExamplePedestrianProfiles.child_male.value, 0.25)
                 ),
                 camera_distances_distributions: Iterable[Iterable[StandardDistribution]] = ((
                     StandardDistribution(-7.0, 2.0),
                     StandardDistribution(0.0, 0.25),
                     StandardDistribution(1.0, 0.25)
                 ),),
                 camera_fov: float = 90.0,
                 camera_image_size: Tuple[int, int] = (800, 600),
                 **kwargs) -> None:
        super().__init__(
            group=kwargs.get('group', None),
            target=kwargs.get('target', None),
            name=kwargs.get('name', None),
            daemon=kwargs.get('daemon', None)
        )

        self._outfile = outfile
        self._outfile_lock = outfile_lock
        self._outputs_dir = outputs_dir
        self._queue = queue

        self._rng = km.KarmaDataProvider.get_rng(seed)

        self._batch_idx = batch_idx
        self._batch_size = batch_size
        self._camera_distances_distributions = camera_distances_distributions
        self._pedestrian_distributions = pedestrian_distributions

        self._clip_length_in_frames = clip_length_in_frames
        self._camera_fov = camera_fov
        self._camera_image_size = camera_image_size

        self._karma = None
        self._kwargs = kwargs

    def run(self) -> None:
        no_of_generated_clips = 0
        with km.Karma(**self._kwargs) as karma:
            self._karma = karma
            map_name = self.get_map_for_batch()
            karma.reset_world(map_name)

            batch_data, no_of_generated_clips = self.generate_batch(map_name)
            with self._outfile_lock:
                pd.DataFrame(batch_data).to_csv(self._outfile, mode='a',
                                                header=(self._batch_idx == 0),
                                                index=False)

        logging.getLogger(__name__).debug(
            f'Generated {no_of_generated_clips} clips.')

        self._queue.put(no_of_generated_clips)

    def get_profiles(self) -> Iterable[Iterable[PedestrianProfile]]:
        """
        Get the pedestrian profiles for each clip.
        By default, this will return a random, single pedestrian profile per clip.
        """
        profiles, weights = zip(*self._pedestrian_distributions)
        indices = self._rng.choice(np.arange(len(profiles)), size=(
            self._batch_size,), replace=True, p=weights)
        return [(profiles[i],) for i in indices]

    def get_spawn_points(self, profiles: Iterable[Iterable[PedestrianProfile]]) -> Iterable[Iterable[carla.Transform]]:
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
                in_clip.append(km.KarmaDataProvider.get_pedestrian_spawn_point())
            spawn_points.append(in_clip)
        return spawn_points

    def get_models(self, profiles: Iterable[Iterable[PedestrianProfile]]) -> Iterable[Iterable[str]]:
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

    def get_pedestrians(self, models: Iterable[Iterable[str]], spawn_points: Iterable[Iterable[carla.Transform]]) -> Iterable[Iterable[km.Walker]]:
        """
        Generate pedestrians for a batch.
        In general, this should not tick the world, but rather just create the pedestrians.
        Karma.tick() will be called after this to actually spawn the pedestrians.

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
                logging.getLogger(__name__).info(
                    f'Failed to create pedestrians for clip {clip_idx} in batch {self._batch_idx}.')
            pedestrians.append(in_clip)
        return pedestrians

    def get_camera_distances(self) -> Iterable[Iterable[float]]:
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

    def get_camera_look_at(self, pedestrians: Iterable[Iterable[km.Walker]], camera_distances: Iterable[Iterable[float]]) -> Iterable[Iterable[carla.Transform]]:
        """
        Get the camera look at points for each clip.
        By default, this will return a random, single camera position per clip.
        """
        return [
            self.get_clip_camera_look_at(clip_idx, clip_pedestrians, clip_camera_distances
                                         ) if len(clip_pedestrians) > 0 else []
            for clip_idx, (clip_pedestrians, clip_camera_distances) in enumerate(zip(pedestrians, camera_distances))
        ]

    def setup_pedestrians(self, pedestrians: Iterable[Iterable[km.Walker]], profiles: Iterable[Iterable[PedestrianProfile]], camera_look_at: Iterable[Iterable[carla.Transform]]) -> None:
        """
        Setup the pedestrians for a batch. It is called after the pedestrians are spawned in the world.
        """
        for clip_idx, (clip_pedestrians, clip_profiles, clip_look_at) in enumerate(zip(pedestrians, profiles, camera_look_at)):
            if len(clip_pedestrians) > 0:
                self.setup_clip_pedestrians(
                    clip_idx, clip_pedestrians, clip_profiles, clip_look_at)

    def get_pedestrians_control(self, pedestrians: Iterable[Iterable[km.Walker]], profiles: Iterable[Iterable[PedestrianProfile]], camera_look_at: Iterable[Iterable[carla.Transform]]) -> Iterable[Iterable[PedestrianControl]]:
        """
        Sets up controllers for all pedestrians.
        :param pedestrians: [description]
        :type pedestrians: Iterable[Iterable[km.Walker]]
        :param profiles: [description]
        :type profiles: Iterable[Iterable[PedestrianProfile]]
        :param camera_look_at: [description]
        :type camera_look_at: Iterable[Iterable[carla.Transform]]
        :return: [description]
        :rtype: Iterable[Iterable[PedestrianControl]]
        """
        return [
            self.get_clip_pedestrians_control(clip_idx, clip_pedestrians, clip_profiles, clip_look_at
                                              ) if len(clip_pedestrians) > 0 else []
            for clip_idx, (clip_pedestrians, clip_profiles, clip_look_at) in enumerate(zip(pedestrians, profiles, camera_look_at))
        ]

    def get_clip_pedestrians_control(self, clip_idx: int, pedestrians: Iterable[km.Walker], profiles: Iterable[PedestrianProfile], camera_look_at: Iterable[carla.Transform]) -> Iterable[PedestrianControl]:
        """
        Get the pedestrians controls for a single clip.
        """
        raise NotImplementedError()

    def setup_clip_pedestrians(self, clip_idx: int, pedestrians: Iterable[km.Walker], profiles: Iterable[PedestrianProfile], camera_look_at: Iterable[carla.Transform]) -> None:
        """
        Setup the pedestrians in a single clip.
        """
        raise NotImplementedError()

    def get_clip_camera_look_at(self, clip_idx: int, pedestrians: Iterable[km.Walker], camera_distances: Iterable[float]) -> Iterable[carla.Transform]:
        """
        Get the camera look at points for a single clip.
        """
        raise NotImplementedError()

    def get_camera_managers(
        self,
        camera_look_at: Iterable[Iterable[carla.Transform]],
        camera_distances: Iterable[Iterable[float]],
        pedestrians: Iterable[Iterable[km.Walker]]
    ) -> Iterable[Iterable[CamerasManager]]:
        """
        Get camera managers for batch. By default this will create a single
        CameraManager per clip, merging images from all specified clip cameras.
        This method ticks the world with each CameraManager created,
        since cameras have to be spawned to attach event listeners.

        :param camera_look_at: [description]
        :type camera_look_at: Iterable[Iterable[carla.Transform]]
        :param camera_distances: [description]
        :type camera_distances: Iterable[Iterable[float]]
        """
        managers = []
        for clip_idx, (clip_look_at, clip_distances, clip_pedestrians) in enumerate(zip(camera_look_at, camera_distances, pedestrians)):
            managers.append(self.get_clip_camera_managers(clip_idx, clip_look_at, clip_distances, clip_pedestrians
                                                          ))
        return managers

    def get_clip_camera_managers(self, clip_idx: int, camera_look_at: Iterable[carla.Transform], camera_distances: Iterable[float], pedestrians: Iterable[km.Walker]) -> Iterable[CamerasManager]:
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

    def get_map_for_batch(self) -> str:
        """
        Get the map for a batch. All pedestrians in batch will be spawned in the same world at the same time.
        """
        return self._rng.choice(km.KarmaDataProvider.get_available_maps())

    def generate_batch(self, map_name: str) -> Iterable[Dict]:
        """
        Generate a single batch.
        """
        profiles = self.get_profiles()
        spawn_points = self.get_spawn_points(profiles)
        models = self.get_models(profiles)
        pedestrians = self.get_pedestrians(models, spawn_points)

        # spawn pedestrians in world
        self._karma.tick()

        # if no pedestrians, skip batch
        if sum(len(pedestrians_per_clip) for pedestrians_per_clip in pedestrians) == 0:
            logging.getLogger(__name__).info(
                f'No pedestrians spawned in batch {self._batch_idx}, skipping.')
            return [], 0

        camera_distances = self.get_camera_distances()
        camera_look_at = self.get_camera_look_at(
            pedestrians, camera_distances)
        camera_managers = self.get_camera_managers(
            camera_look_at, camera_distances, pedestrians)
        # no need to tick, CamerasManager already did it

        self.setup_pedestrians(pedestrians, profiles, camera_look_at)

        # apply settings to pedestrians
        self._karma.tick()

        controllers = self.get_pedestrians_control(
            pedestrians, profiles, camera_look_at)
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
                recording_name = f'{self._batch_idx}-{clip_idx}-{manager_idx}'
                clip_recordings.append(recording_name)
                manager.start_recording(recording_name)
            recordings.append(clip_recordings)

        # prepare data capture callback
        frame_data = [[] for _ in range(self._batch_size)]
        capture_callback_id = self._karma.register_callback(
            km.KarmaStage.tick, lambda snapshot: self.capture_frame_data(snapshot, pedestrians, frame_data))

        # move simulation forward the required number of frames
        # and capture per-frame data
        for _ in trange(self._clip_length_in_frames, desc='Frame', position=1, leave=False):
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
        reached_first_waypoint = []
        for clip_controllers in controllers:
            clip_reached_first_waypoint = []
            for controller in clip_controllers:
                self._karma.unregister_controller(controller)
                clip_reached_first_waypoint.append(controller.reached_first_waypoint)
            reached_first_waypoint.append(
                len(clip_reached_first_waypoint) > 0 and all(clip_reached_first_waypoint))

        # collect batch data
        batch_data, clips_count = self.collect_batch_data(
            map_name, profiles, spawn_points, models, pedestrians, camera_managers, recordings, recorded_frames, frame_data, reached_first_waypoint)

        return batch_data, clips_count

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

                world_pose, component_pose, relative_pose = get_pedestrian_pose_dicts(
                    pedestrian)

                current_frame_data.append({
                    'world.frame': snapshot.frame,
                    'frame.pedestrian.id': pedestrian.id,
                    'frame.pedestrian.transform': convert_transform_to_list(pedestrian_transform),
                    'frame.pedestrian.velocity': convert_vector3d_to_list(pedestrian_velocity),
                    'frame.pedestrian.pose.world': convert_pose_dict_to_list(world_pose),
                    'frame.pedestrian.pose.component': convert_pose_dict_to_list(component_pose),
                    'frame.pedestrian.pose.relative': convert_pose_dict_to_list(relative_pose),
                })

            clip_data.append(current_frame_data)

    def collect_batch_data(self, map_name, profiles, spawn_points, models, pedestrians, camera_managers, recordings, recorded_frames, captured_data, reached_first_waypoint) -> Tuple[List[Dict], int]:
        batch_data = []
        clips_count = 0
        for clip_idx in range(self._batch_size):
            if not sum(len(c) for c in captured_data[clip_idx]):
                logging.getLogger(__name__).info(
                    f'No data was captured for clip {clip_idx}, skipping.')
                continue
            if not reached_first_waypoint[clip_idx]:
                logging.getLogger(__name__).info(
                    f'At least one of the pedestrians in {clip_idx} did not reach first waypoint, skipping.')
                continue

            clip_managers: Iterable[CamerasManager] = camera_managers[clip_idx]
            clip_models: Iterable[str] = models[clip_idx]
            clip_profiles: Iterable[PedestrianProfile] = profiles[clip_idx]
            clip_spawn_points: Iterable[carla.Transform] = spawn_points[clip_idx]
            clip_pedestrians: Iterable[km.Walker] = pedestrians[clip_idx]

            clip_captured_data: Iterable[Dict] = sorted(
                captured_data[clip_idx], key=lambda x: x[0]['world.frame'])
            clip_captured_data_frames: Iterable[int] = [
                data[0]['world.frame']
                for data in clip_captured_data
            ]

            clip_id = str(uuid4())  # make it unique in dataset
            skipped = False

            for manager_idx, manager in enumerate(clip_managers):
                recording: str = recordings[clip_idx][manager_idx] if len(
                    recordings[clip_idx]) else None

                if recording is None:
                    logging.getLogger(__name__).info(
                        f'No recording was created for clip {clip_idx} camera {manager_idx}, skipping.')
                    skipped = True
                    continue

                clip_recorded_frames: Iterable[int] = sorted(
                    recorded_frames[clip_idx][manager_idx])

                for camera_idx, camera in enumerate(manager.get_streamed_cameras()):
                    # TODO: Some of the cameras can move (e.g. attached to a vehicle), so camera position should be per frame
                    # but for now we assume only FreeCameras that are static
                    camera_transform = camera.get_transform()

                    min_frame = clip_recorded_frames[0]
                    max_frame = clip_recorded_frames[-1]

                    skip = 0
                    for frame_idx, world_frame in enumerate(range(min_frame, max_frame + 1)):
                        # skip frame if it is not in the recorded frames - we do not want to have
                        # indexing differences between video and captured data
                        if world_frame not in clip_recorded_frames:
                            skip += 1
                            continue

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
                            assert frame[pedestrian_idx]['frame.pedestrian.id'] == pedestrian.id, 'Pedestrian ID mismatch'
                            assert frame[pedestrian_idx]['world.frame'] == world_frame, 'World frame mismatch'

                            full_frame_data = {
                                'id': clip_id,
                                'world.map': map_name,
                                'camera.idx': camera_idx,
                                'camera.recording': f'clips/{recording}-{camera_idx}.mp4',
                                'camera.transform': convert_transform_to_list(camera_transform),
                                'pedestrian.idx': pedestrian_idx,
                                'pedestrian.model': model,
                                'pedestrian.age': profile.age,
                                'pedestrian.gender': profile.gender,
                                'pedestrian.spawn_point': convert_transform_to_list(spawn_point),
                                'frame.idx': frame_idx - skip,
                            }
                            full_frame_data.update(frame[pedestrian_idx])

                            if 'frame.pedestrian.pose.world' in frame[pedestrian_idx]:
                                camera_pose = project_pose(
                                    convert_list_to_pose_dict(
                                        frame[pedestrian_idx]['frame.pedestrian.pose.world']),
                                    camera_transform,
                                    camera
                                )
                                full_frame_data['frame.pedestrian.pose.camera'] = convert_pose_2d_dict_to_list(
                                    camera_pose)

                            del full_frame_data['frame.pedestrian.id']

                            batch_data.append(full_frame_data)
            if not skipped:
                clips_count += 1
        return batch_data, clips_count


class Generator(object):
    """
    Common class for various datasets generation. Handles managing/saving etc.
    """
    batch_generator = BatchGenerator

    def __init__(self,
                 outputs_dir: str = './datasets',
                 number_of_clips: int = 512,
                 clip_length_in_frames: int = 600,
                 pedestrian_distributions: Iterable[Tuple[PedestrianProfile, float]] = (
                     (ExamplePedestrianProfiles.adult_female.value, 0.25),
                     (ExamplePedestrianProfiles.adult_male.value, 0.25),
                     (ExamplePedestrianProfiles.child_female.value, 0.25),
                     (ExamplePedestrianProfiles.child_male.value, 0.25)
                 ),
                 camera_distances_distributions: Iterable[Iterable[StandardDistribution]] = ((
                     StandardDistribution(-7.0, 2.0),
                     StandardDistribution(0.0, 0.25),
                     StandardDistribution(1.0, 0.25)
                 ),),
                 batch_size: int = 16,
                 camera_fov: float = 90.0,
                 camera_image_size: Tuple[int, int] = (800, 600),
                 **kwargs
                 ) -> None:
        self._outputs_dir = outputs_dir
        # Ensure that the output directory exists AND is empty
        os.makedirs(self._outputs_dir, exist_ok=False)

        # handle complex config data
        self._camera_distances_distributions = self.__parse_camera_position_distributions(
            camera_distances_distributions)
        self._pedestrian_distributions = self.__parse_pedestrian_distributions(
            pedestrian_distributions)

        self._number_of_clips = number_of_clips
        self._clip_length_in_frames = clip_length_in_frames
        self._batch_size = batch_size
        self._camera_fov = camera_fov
        self._camera_image_size = camera_image_size

        self._total_batches = np.ceil(
            self._number_of_clips/self._batch_size).astype(int)
        self._kwargs = kwargs

        mp.set_start_method('spawn')

    def __parse_camera_position_distributions(self, camera_position_distributions):
        converted_camera_position_distributions = []
        for i, camera in enumerate(camera_position_distributions):
            if len(camera) != 3:
                raise ValueError(
                    f'Camera position distribution {i} should have exactly 3 elements (for x,y,z), got {len(distribution)}')
            camera_distributions = []
            for distribution in camera:
                if isinstance(distribution, StandardDistribution):
                    camera_distributions.append(distribution)
                else:
                    camera_distributions.append(StandardDistribution(*distribution))
            converted_camera_position_distributions.append(camera_distributions)
        return converted_camera_position_distributions

    def __parse_pedestrian_distributions(self, pedestrian_distributions):
        converted_pedestrian_distributions = []
        for distribution in pedestrian_distributions:
            profile, weight = distribution
            if isinstance(profile, PedestrianProfile):
                converted_pedestrian_distributions.append((profile, weight))
            elif isinstance(profile, str):
                converted_pedestrian_distributions.append(
                    (ExamplePedestrianProfiles[profile].value, weight))
            else:
                assert len(
                    profile) == 4, f'Pedestrian profile should have exactly 4 elements, got {len(profile)}'
                (age, gender, (walking_speed_mean, walking_speed_std),
                 (crossing_speed_mean, crossing_speed_std)) = profile
                converted_pedestrian_distributions.append(
                    (PedestrianProfile(
                        age, gender,
                        StandardDistribution(walking_speed_mean, walking_speed_std),
                        StandardDistribution(crossing_speed_mean, crossing_speed_std)
                    ), weight))
        return converted_pedestrian_distributions

    @staticmethod
    def add_cli_args(parser):
        subparser = parser.add_argument_group('Generator')

        subparser.add_argument('--outputs-dir', default=None, type=str,
                               help='Directory to store outputs (default: ./datasets).')
        subparser.add_argument('--number-of-clips', type=int, default=512,
                               help='Total number of clips to generate.')
        subparser.add_argument('--clip-length-in-frames', type=int, default=600,
                               help='Length of each clip in frames.')
        subparser.add_argument('--batch-size', type=int, default=16,
                               help='Number of clips in each batch.')
        subparser.add_argument('--camera-fov', type=float, default=90.0,
                               help='Camera horizontal FOV in degrees.')
        subparser.add_argument('--camera-image-size', type=ast.literal_eval, default='(800,600)',
                               help='Camera image size in pixels as a (width, height) tuple (default: (800,600)).')

        return parser

    def generate(self) -> None:
        """
        Generate the dataset.
        """

        outfile = os.path.join(self._outputs_dir, 'data.csv')
        outfile_lock = mp.Lock()
        results_queue = mp.Queue()

        generated_clips = []
        batch_idx = 0
        failed = 0
        server_failed = 0

        with tqdm(total=self._number_of_clips, desc='Clips', position=0, postfix={'failed': 0}) as pbar:
            while sum(generated_clips) < self._number_of_clips and batch_idx < 2*self._total_batches:
                failed = batch_idx - sum(generated_clips)
                pbar.set_postfix(failed=failed)

                # TODO: this has the potential for a 'real' multiprocessing if multiple servers are available
                seed = self._kwargs.get('seed', None)
                batch_generation_process = self.batch_generator(
                    outfile=outfile,
                    outfile_lock=outfile_lock,
                    outputs_dir=self._outputs_dir,
                    queue=results_queue,
                    batch_idx=batch_idx,
                    batch_size=self._batch_size,
                    clip_length_in_frames=self._clip_length_in_frames,
                    pedestrian_distributions=self._pedestrian_distributions,
                    camera_distances_distributions=self._camera_distances_distributions,
                    camera_fov=self._camera_fov,
                    camera_image_size=self._camera_image_size,
                    **{
                        **self._kwargs,
                        'seed': seed + batch_idx if seed is not None else None
                    }
                )
                batch_generation_process.start()
                batch_generation_process.join()

                batch_idx += 1

                if results_queue.empty():
                    logging.getLogger(__name__).warning(
                        f'Process failed for batch {batch_idx}')
                    server_failed += 1
                    # assume that the server will restart
                    time.sleep(float(os.getenv('CARLA_SERVER_START_PERIOD', '30.0')))
                    continue

                generated_clips.append(results_queue.get())

                if generated_clips[-1] > 0:
                    pbar.update(generated_clips[-1])

        logging.getLogger(__name__).info(
            f'Generated {sum(generated_clips)} clips out of desired {self._number_of_clips}. Batch generation failed {failed} times, including {server_failed} server failures/timeouts.')
