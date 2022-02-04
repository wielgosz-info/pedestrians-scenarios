from enum import Enum, auto
from typing import Tuple, Union

import carla
import numpy as np
from PIL import Image

from ..karma_data_provider import KarmaDataProvider


class CaptureFailureMode(Enum):
    none = auto()
    zero = auto()
    noise = auto()
    last = auto()


class Camera(object):
    def __init__(self,
                 sensor: carla.Sensor,
                 capture_failure_mode: CaptureFailureMode = CaptureFailureMode.zero,
                 register: bool = True,
                 **kwargs
                 ):
        if not sensor.type_id.startswith('sensor.camera'):
            raise ValueError('Sensor must be a camera.')

        self._camera = sensor

        # TODO: add support for other camera types
        self._type = self._camera.type_id.replace('sensor.camera.', '')
        self._converter = {
            'rgb': carla.ColorConverter.Raw,
            'depth': carla.ColorConverter.LogarithmicDepth if kwargs.get('logarithmic', False) else carla.ColorConverter.Depth,
            'semantic_segmentation': carla.ColorConverter.CityScapesPalette,
        }[self._type]

        self._image_size = (
            int(self._camera.attributes['image_size_x']),
            int(self._camera.attributes['image_size_y'])
        )
        self._image_shape = (self._image_size[1], self._image_size[0], 3)

        self._capture_failure_mode = capture_failure_mode
        self._last_frame = None
        if self._capture_failure_mode == 'last':
            self._last_frame = np.zeros(self._image_shape, dtype=np.uint8)

        if register:
            KarmaDataProvider.register_sensor_queue(self._camera)

    @property
    def image_size(self) -> Tuple[int]:
        return self._image_size

    @property
    def sensor(self) -> carla.Sensor:
        return self._camera

    def get_data(self) -> Union[np.ndarray, None]:
        """
        Returns the most recent image from camera as an RGB numpy array (PIL-compatible).
        Depending on settings, can return None, zeros array, random noise or repeated last frame
        if no image is available.

        :return: [description]
        :rtype: Union[np.ndarray, None]
        """
        frames = KarmaDataProvider.get_sensor_data(self._camera.id)

        if len(frames):
            data = frames[-1]
            data.convert(self._converter)
            img = Image.frombuffer('RGBA', (data.width, data.height),
                                   data.raw_data, 'raw', 'RGBA', 0, 1)  # load
            img = img.convert('RGB')  # drop alpha
            # the data is actually in BGR format, so switch channels
            self._last_frame = np.array(img)[..., ::-1]
        else:
            if self._capture_failure_mode == CaptureFailureMode.zero:
                self._last_frame = np.zeros(self._image_shape, dtype=np.uint8)
            elif self._capture_failure_mode == CaptureFailureMode.noise:
                self._last_frame = KarmaDataProvider.get_rng().randint(
                    0, 255, size=self._image_shape, dtype=np.uint8)
            elif self._capture_failure_mode == CaptureFailureMode.last:
                self._last_frame = self._last_frame.copy()
            else:
                self._last_frame = None

        return self._last_frame.copy()
