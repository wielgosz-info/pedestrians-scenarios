import os
from typing import Iterable, Union
import numpy as np
from .segmentation_renderer import SegmentationRenderer
from .dvs_palette import DYNAMIC_VISION_SENSOR_PALETTE
from PIL import Image
from enum import Enum, auto


class DVSOutputFormat(Enum):
    img_frame = auto()
    img_timestamp = auto()
    events = auto()


class DVSRenderer(SegmentationRenderer):
    def __init__(
        self,
        palette=DYNAMIC_VISION_SENSOR_PALETTE,
        mode: DVSOutputFormat = DVSOutputFormat.img_frame,
        **kwargs
    ) -> None:
        super().__init__(palette=palette, **kwargs)

        self._mode = mode

    def render_clip(self, clip: Iterable[np.ndarray]) -> np.ndarray:
        if self._mode == DVSOutputFormat.img_frame:
            frames = np.zeros(
                (len(clip), self.image_size[1], self.image_size[0]),
                dtype=np.uint8
            )
            for frame_idx, events in enumerate(clip):
                frames[frame_idx, events[:]['y'], events[:]['x']
                       ] = events[:]['pol'] + 1  # +1 to avoid 0
            return super().render_clip(frames)

        events = np.concatenate(clip, axis=0)
        min_ts = max(events['t'].min() - 1, 0)
        max_ts = events['t'].max() + 1

        # adjust timestamps to start from 1
        events['t'] -= min_ts

        # tmp sanity check - max timestamp should be less than clip length * (1000 / fps)
        assert max_ts < len(clip) * (1000 / 30.0) + 1

        if self._mode == DVSOutputFormat.events:
            # sort in ascending order by timestamp
            events.sort(axis=0, order=['t', 'x', 'y'])
            return events

        frames = np.zeros(
            (max_ts - min_ts, self.image_size[1], self.image_size[0]),
            dtype=np.uint8
        )
        frames[
            events['t'],
            events['y'],
            events['x']
        ] = events['pol'] + 1  # +1 to avoid 0
        return super().render_clip(frames)

    def save(self, frames: Union[Iterable[Image.Image], np.ndarray], name: str = 'out', outputs_dir: str = None, fps: int = 30) -> None:
        if outputs_dir is None:
            outputs_dir = os.path.join(os.getcwd(), 'dvs_renderer')

        if self._mode != DVSOutputFormat.events:
            fps = fps if self._mode == DVSOutputFormat.img_frame else 1000
            return super().save(frames, name, outputs_dir, fps)
        else:
            return np.save(os.path.join(outputs_dir, name), frames, allow_pickle=False)
