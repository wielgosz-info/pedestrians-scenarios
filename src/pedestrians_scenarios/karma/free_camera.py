from typing import List, Tuple, Union

import carla
import numpy as np
from PIL import Image

from .utils.rotations import look_at as carla_look_at

from .actor import Actor
from .karma_data_provider import KarmaDataProvider


class FreeCamera(Actor):
    def __init__(self,
                 look_at: carla.Transform,
                 distance: Tuple[float] = (-10.0, 0.0, 0.0),
                 image_size: Tuple[int] = (800, 600),
                 fov: float = 90.0,
                 data_failure_mode: Union['none', 'zero', 'noise', 'last'] = 'zero',
                 **kwargs
                 ):
        camera_location = carla.Location(look_at.transform(carla.Location(*distance)))
        camera_transform = carla_look_at(look_at, camera_location)

        blueprint_library = KarmaDataProvider.get_blueprint_library()
        camera_bp = blueprint_library.find('sensor.camera.rgb')

        camera_bp.set_attribute('image_size_x', str(image_size[0]))
        camera_bp.set_attribute('image_size_y', str(image_size[1]))
        camera_bp.set_attribute('fov', str(fov))

        self.__image_size = image_size
        self.__image_shape = (self.__image_size[1], self.__image_size[0], 3)
        camera = KarmaDataProvider.request_new_sensor(
            camera_bp, camera_transform, **kwargs)
        super().__init__(actor=camera)

        self.__data_failure_mode = data_failure_mode
        self.__last_frame = None
        if self.__data_failure_mode == 'last':
            self.__last_frame = np.zeros(self.__image_shape, dtype=np.uint8)

    @property
    def image_size(self) -> Tuple[int]:
        return self.__image_size

    def get_data(self) -> Union[np.ndarray, None]:
        """
        Returns the most recent image from camera as an RGB numpy array (PIL-compatible).
        Depending on settings, can return None, zeros array, random noise or repeated last frame
        if no image is available.

        :return: [description]
        :rtype: Union[np.ndarray, None]
        """
        frames = KarmaDataProvider.get_sensor_data(self.id)

        if len(frames):
            data = frames[-1]
            data.convert(carla.ColorConverter.Raw)
            img = Image.frombuffer('RGBA', (data.width, data.height),
                                   data.raw_data, 'raw', 'RGBA', 0, 1)  # load
            img = img.convert('RGB')  # drop alpha
            # the data is actually in BGR format, so switch channels
            self.__last_frame = np.array(img)[..., ::-1]
        else:
            if self.__data_failure_mode == 'zero':
                self.__last_frame = np.zeros(self.__image_shape, dtype=np.uint8)
            elif self.__data_failure_mode == 'noise':
                self.__last_frame = KarmaDataProvider.get_rng().randint(
                    0, 255, size=self.__image_shape, dtype=np.uint8)
            elif self.__data_failure_mode == 'last':
                self.__last_frame = self.__last_frame.copy()
            else:
                self.__last_frame = None

        return self.__last_frame.copy()
