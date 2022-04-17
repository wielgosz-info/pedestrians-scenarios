import pandas as pd
import os
import shutil
import re
from tqdm.auto import tqdm


if __name__ == '__main__':
    input_dirs = [
        "/datasets/CARLA/BasicPedestriansCrossing",
        "/datasets/CARLA/BasicPedestriansCrossing_jitter-10",
    ]

    output_dir = "/app/BasicPedestriansCrossing_v2"
    clips_dir = os.path.join(output_dir, 'clips')

    os.makedirs(clips_dir)

    dfs = []
    name_regex = re.compile(
        r'(?P<batch_idx>\d+)-(?P<clip_idx>\d+)-(?P<pedestrian_idx>\d+)-(?P<camera_idx>\d+)\.mp4')

    for in_dir in tqdm(input_dirs):
        df = pd.read_csv(in_dir + "/data.csv")
        clip_groups = df.groupby(by=['id', 'camera.recording'])
        for (clip_id, clip_path), index in tqdm(clip_groups.groups.items()):
            name_regex_match = name_regex.match(os.path.basename(clip_path))
            if name_regex_match:
                new_name = '{}-{}-{}.mp4'.format(
                    clip_id,
                    name_regex_match.group('pedestrian_idx'),
                    name_regex_match.group('camera_idx'),
                )
                new_path = os.path.join(clips_dir, new_name)
                shutil.copyfile(os.path.join(in_dir, clip_path), new_path)
                df.loc[index, 'camera.recording'] = os.path.join('clips', new_name)
        dfs.append(df)

    df = pd.concat(dfs)
    df.to_csv(os.path.join(output_dir, 'data.csv'), index=False, header=True)
