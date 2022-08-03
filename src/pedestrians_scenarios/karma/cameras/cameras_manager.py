import itertools
import os
import uuid
from enum import Enum, auto
from typing import Dict, Iterable, List, Tuple, Union

import av
import carla
import numpy as np
from PIL import Image

from pedestrians_scenarios.karma.renderers.segmentation_renderer import SegmentationRenderer

from ..karma import Karma, KarmaStage
from ..karma_data_provider import KarmaDataProvider
from .camera import Camera
from .free_camera import FreeCamera


class FramesMergingMathod(Enum):
    none = auto()
    square = auto()
    horizontal = auto()
    vertical = auto()


class CamerasManager(object):
    def __init__(self,
                 karma: Karma,
                 outputs_dir: str,
                 merging_method: Union[FramesMergingMathod,
                                       Tuple[int, int]] = FramesMergingMathod.square,
                 ):
        self.__karma = karma

        self.__outputs_dir = outputs_dir
        os.makedirs(self.__outputs_dir, exist_ok=True)

        # managed cameras are those that Cameras Manager is responsible for (creation and deletion)
        self.__managed_cameras: Dict[int, FreeCamera] = {}
        # streamed cameras are all cameras that the Camera Manager should get data from
        # this includes managed cameras and e.g. vehicle cameras
        self.__streamed_cameras: Dict[int, Union[FreeCamera, Camera]] = {}
        # some cameras are synchronized, i.e. they show the same scene, but in different 'modes'
        self.__synchronized_cameras: List[List[int]] = []

        if not isinstance(merging_method, FramesMergingMathod):
            assert len(
                merging_method) == 2, "merging_method must be video columns, video rows tuple"
            merging_method = (*merging_method, 1)  # save to a single file

        self.__merging_method = merging_method
        self.__containers = []
        self.__streams = []
        self.__video_columns = None
        self.__video_rows = None
        self.__column_width = None
        self.__row_height = None
        self.__files_number = None
        self.__tick_callback_id = None
        self.__close_callback_id = None
        self.__names = []

        self.__segmentation_labels = []
        self.__segmentation_renderer = SegmentationRenderer()

        self.__captured_frames: List[int] = None

    @property
    def outputs_dir(self) -> str:
        return self.__outputs_dir

    def create_free_cameras(self,
                            cameras: Iterable[Tuple[carla.Transform, Iterable[float]]],
                            image_size: Tuple[int, int] = (800, 600),
                            fov: float = 90.0,
                            camera_types: Iterable[str] = (
                                'rgb', 'semantic_segmentation'),
                            ):
        """
        Creates cameras at the given locations and registers them with the Camera Manager.
        This forcibly ticks the world, because cameras need to be spawned to attach listeners.

        :param cameras: List of carla.Transform look_at, (x, y, z) distance tuples
        :type cameras: List[Tuple[carla.Transform, Tuple[float]]]
        :param image_size: Size of the camera image (width, height).
            Currently all cameras need to have the same dimensions. Default: (800, 600)
        :type image_size: Tuple[int, int]
        :param fov: Field of view of the camera in degrees. Default: 90.0
        :type fov: float
        :param camera_types: List of camera types to create. Default: ('rgb', 'semantic_segmentation')
        :type camera_types: Iterable[str]
        """
        for look_at, distance in cameras:
            synced = []
            for camera_type in camera_types:
                camera = FreeCamera(
                    look_at=look_at,
                    distance=distance,
                    image_size=image_size,
                    fov=fov,
                    tick=False,
                    register=False,
                    camera_type=camera_type,
                )
                self.__managed_cameras[camera.id] = camera
                synced.append(camera.id)
            if len(synced) > 1:
                self.__synchronized_cameras.append(synced)

        # spawn cameras
        self.__karma.tick()

        # register cameras with KarmaDataProvider (listeners)
        for camera in self.__managed_cameras.values():
            KarmaDataProvider.register_sensor_queue(camera.sensor)

        # register cameras for streaming
        for camera in self.__managed_cameras.values():
            self.register_streamed_camera(camera)

    def register_streamed_camera(self, camera: Union[FreeCamera, Camera, carla.Sensor]):
        """
        Adds a camera to the list of cameras that the Camera Manager should get data from.
        Assumption: camera life cycle is managed by the caller.

        If camera is a carla.Sensor, it will be wrapped in a Camera object,
        which will try to register the data queue for it with
        KarmaDataProvider.register_sensor_queue().

        Assumptions: camera already is spawned it the world, camers other than
        carla.Sensor are assumed to be already registered with KarmaDataProvider.

        :param camera: Karma.FreeCamera or CARLA camera-like Sensor
        :type camera: Union[FreeCamera, carla.Sensor]
        """
        if self.__containers is not None and len(self.__containers):
            raise RuntimeError(
                'Recording session is in progress, all cameras must be registered before recording starts.')

        if isinstance(camera, carla.Sensor):
            camera = Camera(sensor=camera)

        self.__streamed_cameras[camera.id] = camera
        self.__captured_frames = []

    def get_merging_info(self):
        no_streamed_cameras = len(self.__streamed_cameras)
        files = 1

        if no_streamed_cameras < 3 and self.__merging_method == FramesMergingMathod.square:
            self.__merging_method = FramesMergingMathod.horizontal

        if self.__merging_method == FramesMergingMathod.none:
            # special case, each camera is a separate video
            video_columns = 1
            video_rows = 1
            files = no_streamed_cameras
        elif self.__merging_method == FramesMergingMathod.square:
            # find how many videos we need per row and column
            video_columns = int(np.ceil(np.sqrt(no_streamed_cameras)))
            video_rows = int(
                np.ceil(no_streamed_cameras / video_columns))
        elif self.__merging_method == FramesMergingMathod.vertical:
            video_columns = 1
            video_rows = no_streamed_cameras
        elif self.__merging_method == FramesMergingMathod.horizontal:
            video_columns = no_streamed_cameras
            video_rows = 1
        else:
            video_columns, video_rows, files = self.__merging_method

        return video_columns, video_rows, files

    def start_recording(self, session_id: str = None) -> Iterable[str]:
        """
        Starts recording stream for all cameras.
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        self.__video_columns, self.__video_rows, self.__files_number = self.get_merging_info()

        all_cameras: List[Union[FreeCamera, Camera]] = list(
            self.__streamed_cameras.values())
        sizes = np.array([camera.image_size for camera in all_cameras])
        cameras_per_file = self.__video_rows*self.__video_columns

        self.__column_width = sizes[:, 0].max()
        self.__row_height = sizes[:, 1].max()
        self.__total_width = self.__column_width * self.__video_columns
        self.__total_height = self.__row_height * self.__video_rows

        names = []
        for fi in range(self.__files_number):
            name = '{}-{}'.format(session_id, fi)
            names.append(name)
            container = av.open(os.path.join(self.__outputs_dir,
                                             f'{name}.mp4'), mode="w")
            stream = container.add_stream('libx264', rate=30)
            stream.width = self.__total_width
            stream.height = self.__total_height
            stream.pix_fmt = "yuv420p"
            stream.options = {}

            self.__containers.append(container)
            self.__streams.append(stream)

            # we also want to save unprocessed segmentation labels if there are any
            segmentation_labels = {}
            cameras = all_cameras[fi*cameras_per_file:(fi+1)*cameras_per_file]
            for camera in cameras:
                if camera.camera_type == 'semantic_segmentation':
                    segmentation_labels[camera.id] = []

            self.__segmentation_labels.append(segmentation_labels)

        self.__tick_callback_id = self.__karma.register_callback(
            KarmaStage.tick, self.__on_tick)
        self.__close_callback_id = self.__karma.register_callback(
            KarmaStage.close, self.stop_recording)

        self.__names = names
        return names

    def __on_tick(self, snapshot: carla.WorldSnapshot, *args, **kwargs):
        """
        Callback called during Karma.on_carla_tick.
        """

        all_cameras: List[Union[FreeCamera, Camera]] = list(
            self.__streamed_cameras.values())
        cameras_per_file = self.__video_rows*self.__video_columns

        # TODO: each camera should have a save format (MP4 or APNG) and the division should be based on that
        # instead of directly using camera_type and saving segmentation twice
        for fi in range(len(self.__containers)):
            container = self.__containers[fi]
            stream = self.__streams[fi]
            img = np.zeros((self.__total_height, self.__total_width, 3), dtype=np.uint8)
            cameras = all_cameras[fi*cameras_per_file:(fi+1)*cameras_per_file]

            for (r, c), camera in zip(
                itertools.product(range(self.__video_rows),
                                  range(self.__video_columns)),
                cameras
            ):
                data = camera.get_data()
                img[
                    r*self.__row_height:r*self.__row_height+data.shape[0],
                    c*self.__column_width:c*self.__column_width+data.shape[1],
                    :
                ] = data

                if camera.camera_type == 'semantic_segmentation':
                    # get_raw_data never returns anything if data was not captured
                    labels = camera.get_raw_data()
                    if labels is not None:
                        # CARLA encodes labels in red channel & raw data is in BGRA format
                        self.__segmentation_labels[fi][camera.id].append(labels[..., 2])
                    else:
                        # return all zeros (Unknown)
                        self.__segmentation_labels[fi][camera.id].append(
                            np.zeros((data.shape[0], data.shape[1]), dtype=np.uint8))

            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            frame.pict_type = "NONE"

            try:
                for packet in stream.encode(frame):
                    container.mux(packet)

                self.__captured_frames.append(snapshot.frame)
            except av.error.EOFError:
                pass  # recording already finished

    def stop_recording(self):
        """
        Finishes recording stream for all cameras.
        """
        if self.__containers is not None:
            for fi in range(len(self.__containers)):
                # Flush stream
                for packet in self.__streams[fi].encode():
                    self.__containers[fi].mux(packet)

                self.__containers[fi].close()

        if self.__segmentation_labels is not None:
            for fi, name in enumerate(self.__names):
                if not self.__segmentation_labels[fi]:
                    continue

                file_labels = np.stack(
                    list(self.__segmentation_labels[fi].values()), axis=0)
                if file_labels.shape[0] > 1:
                    for ci, sequence in enumerate(self.__segmentation_renderer.render(file_labels)):
                        self.__segmentation_renderer.save(
                            sequence, f'{name}-{ci}', self.__outputs_dir)
                else:
                    sequence = next(self.__segmentation_renderer.render(file_labels))
                    self.__segmentation_renderer.save(
                        sequence, name, self.__outputs_dir)

                # remove obsolete segmentation mp4 file
                os.remove(os.path.join(self.__outputs_dir, f'{name}.mp4'))

        if self.__tick_callback_id is not None:
            self.__karma.unregister_callback(self.__tick_callback_id)

        if self.__close_callback_id is not None:
            self.__karma.unregister_callback(self.__close_callback_id)

        captured = self.__captured_frames

        self.__containers = []
        self.__streams = []
        self.__segmentation_labels = []
        self.__files_number = None
        self.__video_columns = None
        self.__video_rows = None
        self.__column_width = None
        self.__row_height = None
        self.__captured_frames = None

        return captured

    def get_streamed_cameras(self) -> List[Camera]:
        """
        Returns a list of cameras that are being streamed.
        """
        return list(self.__streamed_cameras.values())

    def get_synchronized_cameras(self) -> List[List[Camera]]:
        """
        Returns a list of cameras that are synchronized.
        If there is no synchronization, returns empty list.
        """
        return [
            [self.__streamed_cameras[camera_id] for camera_id in sync_group]
            for sync_group in self.__synchronized_cameras
        ]
