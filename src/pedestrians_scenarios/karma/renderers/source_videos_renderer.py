import glob
import logging
import math
import os
import warnings
from typing import List, Dict, Any

import numpy as np
import pims
import av
from .renderer import Renderer
from .points_renderer import PointsRenderer


class SourceVideosRenderer(Renderer):
    def __init__(self, data_dir: str, overlay_skeletons: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)

        self.__data_dir = data_dir
        self.__overlay_skeletons = overlay_skeletons

    @property
    def overlay_skeletons(self) -> bool:
        return self.__overlay_skeletons

    def render(self, meta: List[Dict[str, Any]], **kwargs) -> List[np.ndarray]:
        # TODO: bboxes/skeletons should be in targets, not meta?
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        rendered_videos = len(meta['video_id'])

        for clip_idx in range(rendered_videos):
            video = self.render_clip(
                meta['video_id'][clip_idx],
                meta['pedestrian_id'][clip_idx],
                meta['clip_id'][clip_idx],
                meta['start_frame'][clip_idx],
                meta['end_frame'][clip_idx],
                meta['bboxes'][clip_idx] if 'bboxes' in meta else None,
                [{
                    'keypoints': sk['keypoints'][clip_idx],
                    'color': sk['color'],
                    'type': sk['type']
                } for sk in meta['skeletons']] if 'skeletons' in meta else None
            )
            yield video

    def render_clip(self, video_id, pedestrian_id, clip_id, start_frame, end_frame, bboxes=None, skeletons=None):
        (canvas_width, canvas_height) = self._image_size
        half_width = int(math.floor(canvas_width / 2))
        half_height = int(math.floor(canvas_height / 2))
        canvas = np.zeros((end_frame - start_frame, canvas_height,
                          canvas_width, 3), dtype=np.uint8)

        paths = glob.glob(os.path.join(self.__data_dir, '{}.*'.format(os.path.splitext(video_id)[0])))
        try:
            assert len(paths) == 1
            with pims.PyAVReaderIndexed(paths[0]) as video:
                clip = video[start_frame:end_frame]
                (clip_height, clip_width, _) = clip.frame_shape

                if bboxes is None:
                    centers = np.array([(clip_width/2, clip_height/2)] *
                                       (end_frame - start_frame)).round().astype(np.int)
                elif isinstance(bboxes, np.ndarray):
                    centers = (bboxes.mean(axis=-2) + 0.5).round().astype(np.int)
                else:
                    centers = (bboxes.mean(dim=-2) +
                               0.5).round().cpu().numpy().astype(int)

                for idx in range(len(clip)):
                    self.render_frame(canvas[idx], clip[idx],
                                      (half_width, half_height),
                                      centers[idx],
                                      (clip_width, clip_height),
                                      skeletons=[{
                                          'keypoints': np.array(sk['keypoints'][idx], np.int32),
                                          'color': sk['color'],
                                          'type': sk['type']
                                      } for sk in skeletons] if skeletons is not None and self.__overlay_skeletons else None)
        except AssertionError:
            # no video or multiple candidates - skip
            logging.getLogger(__name__).warn(
                "Clip extraction failed for {}, {}, {}".format(video_id, pedestrian_id, clip_id))

        return canvas

    @staticmethod
    def render_frame(canvas, clip, frame_half_size, bbox_center, clip_size, skeletons=None):
        (half_width, half_height) = frame_half_size
        (x_center, y_center) = bbox_center
        (clip_width, clip_height) = clip_size

        frame_x_min = int(max(0, x_center-half_width))
        frame_x_max = int(min(clip_width, x_center+half_width))
        frame_y_min = int(max(0, y_center-half_height))
        frame_y_max = int(min(clip_height, y_center+half_height))
        frame_width = frame_x_max - frame_x_min
        frame_height = frame_y_max - frame_y_min
        canvas_x_shift = max(0, half_width-x_center)
        canvas_y_shift = max(0, half_height-y_center)
        canvas[canvas_y_shift:canvas_y_shift+frame_height, canvas_x_shift:canvas_x_shift +
               frame_width] = clip[frame_y_min:frame_y_max, frame_x_min:frame_x_max]

        if skeletons is not None:
            for skeleton in skeletons:
                SourceVideosRenderer.overlay_skeleton(
                    canvas, skeleton, (canvas_x_shift-frame_x_min, canvas_y_shift-frame_y_min))

        return canvas

    @staticmethod
    def overlay_skeleton(canvas, skeleton, shift=(0, 0)):
        keypoints = skeleton['keypoints']
        skeleton_type = skeleton['type']
        color = skeleton['color']

        shifted_points = keypoints + np.array(shift)

        canvas[:] = PointsRenderer.draw_projection_points(
            canvas,
            shifted_points,
            skeleton_type,
            color_values=[color]*len(skeleton_type) if color is not None else None,
            lines=True
        )

        return canvas

    def frames_to_video(self, frames: np.ndarray, video_name: str = 'out', outputs_dir: str = None, fps: int = 30) -> None:
        if outputs_dir is None:
            outputs_dir = os.path.join(os.getcwd(), 'source_videos_renderer')
        os.makedirs(outputs_dir, exist_ok=True)

        output_filename = os.path.join(outputs_dir, video_name + '.mp4')

        with av.open(output_filename, mode="w") as container:
            stream = container.add_stream('libx264', rate=fps)
            stream.width = self._image_size[0]
            stream.height = self._image_size[1]
            stream.pix_fmt = "yuv420p"
            stream.options = {}

            for img in frames:
                frame = av.VideoFrame.from_ndarray(img, format="rgb24")
                frame.pict_type = "NONE"

                for packet in stream.encode(frame):
                    container.mux(packet)

            for packet in stream.encode():
                container.mux(packet)
