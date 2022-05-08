import ast
import logging
import os
import sys
import cv2
import numpy as np
import urllib.request

import pandas as pd
from tqdm.auto import tqdm


logger = logging.getLogger(__name__)


def add_cli_args(parser):
    parser.add_argument(
        '--dataset_dir',
        type=str,
        help='Path to the dataset directory.'
    )
    parser.add_argument(
        '--remove',
        action='store_true',
        default=False,
        help='Actually remove files with no pedestrians from the folder instead of performing dry run.'
    )
    parser.add_argument(
        '--yolo_root',
        type=str,
        help='Path to the YOLO root directory.',
        default='/outputs/YOLO_v3'
    )
    parser.add_argument(
        '--min_clip_length',
        type=int,
        help='Minimum length of the clip in frames.',
        default=30
    )

    return parser


def command(dataset_dir, yolo_root, remove, min_clip_length, **kwargs):
    """
    Command line interface for cleaning datasets.

    Cleaning is done by:
    a) removing all the clips that are on disk but not in CSV, and
    b) removing all clips in which the pedestrian (as detected by YOLOv3)
       is not visible (e.g. due to obstacles between pedestrian and camera
       that were not detected during clips generation).

    By default only dry run is performed, set --remove flag to actually remove the clips.
    The cleaning is then done **IN PLACE**.

    :param kwargs: parsed command line arguments
    :type kwargs: Dict
    """

    csv_path = os.path.join(dataset_dir, 'data.csv')
    video_path = os.path.join(dataset_dir, 'clips')

    df = trim_useless_rows(csv_path, remove=remove, min_clip_length=min_clip_length)

    # TODO: edit the actual video files and update frame indices in the CSV?

    file_names_common, df = sync_csv_and_videos(csv_path, video_path, remove=remove, df=df)
    file_names_no_pedestrians = pedestrian_detection_in_video(
        video_path, file_names_common, remove=remove, yolo_root=yolo_root)
    sync_csv_and_videos(csv_path, video_path, file_names_no_pedestrians, remove=remove, df=df)


def print_files_info(files_in_dir, files_in_csv,
                     common_files,
                     remove_from_dir, remove_from_csv):
    print('Number of the video files: {}'.format(len(files_in_dir)))
    print('Number of the videos in CSV: {}'.format(len(files_in_csv)))
    print('Number of common files between the dir and the CSV: {}'.format(
        len(common_files)))
    print('Number of files to be removed from the folder: {}'.format(
        len(remove_from_dir)))
    print('Number of files to be removed from the CSV: {}'.format(
        len(remove_from_csv)))


def sync_csv_and_videos(csv_path, video_path, removed_from_dir_dry=None, remove=False, df=None):
    """
    This function makes sure that the videos and the csv are in sync.
    """
    logger.info('Starting CSV and video files synchronization...')

    df = get_df(csv_path, df)

    (files_in_dir, files_in_csv,
     common_files,
     remove_from_dir, remove_from_csv) = get_file_lists(video_path, df, removed_from_dir_dry)

    print_files_info(
        files_in_dir,
        files_in_csv,
        common_files,
        remove_from_dir, remove_from_csv)

    if len(remove_from_dir):
        logger.debug('Will remove from folder:\n' + ('\n'.join(remove_from_dir)))
    if len(remove_from_csv):
        logger.debug('Will remove from CSV:\n' + ('\n'.join(remove_from_csv)))

    if len(remove_from_csv):
        remove_from_csv_prefixed = [os.path.join(
                'clips', element) for element in remove_from_csv]
        df = df[~df['camera.recording'].isin(remove_from_csv_prefixed)]

    if remove:
        if len(remove_from_dir):
            for element in tqdm(remove_from_dir, desc='Removing the files'):
                os.remove(os.path.join(video_path, element))

        if len(remove_from_csv):
            logger.info('Saving updated CSV.')
            df.to_csv(csv_path, index=False)

    logger.info('CSV and video files synchronization done.')

    return common_files, df


def get_file_lists(video_path, df, removed_from_dir_dry=None):
    files_in_dir = set(os.listdir(video_path))
    files_in_csv = set([path.replace('clips/', '')
                        for path in df['camera.recording'].unique()])

    if removed_from_dir_dry is not None:
        files_in_dir = files_in_dir.difference(removed_from_dir_dry)

    # find common elements
    common_files = files_in_dir.intersection(files_in_csv)

    # subtract common elements from the set of file names from the folder and csv
    remove_from_dir = files_in_dir.difference(common_files)
    remove_from_csv = files_in_csv.difference(common_files)

    return files_in_dir, files_in_csv, common_files, remove_from_dir, remove_from_csv


def pedestrian_detection_in_video(video_path, files_list=None, remove=False, yolo_root='/outputs/YOLO_v3'):
    """
    Checks if there is a pedestrian visible in the video in at least one frame.
    Taken from:
    https://github.com/arunponnusamy/object-detection-opencv
    """
    logger.info('Starting pedestrian detection in videos.')

    if files_list is None:
        file_names_from_folder = os.listdir(video_path)
    else:
        file_names_from_folder = files_list

    scale = 0.00392
    yolov3_cfg = os.path.join(yolo_root, 'yolov3.cfg')
    yolov3_weights = os.path.join(yolo_root, 'yolov3.weights')

    os.makedirs(yolo_root, exist_ok=True)

    # check if the files exist if not download them from url and put under current location
    if not os.path.exists(yolov3_cfg):
        logger.info('Downloading yolov3.cfg...')
        urllib.request.urlretrieve(
            'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg',
            yolov3_cfg
        )
    if not os.path.exists(yolov3_weights):
        logger.info('Downloading yolov3.weights...')
        urllib.request.urlretrieve(
            'https://pjreddie.com/media/files/yolov3.weights',
            yolov3_weights
        )

    # read pre-trained model and config file
    net = cv2.dnn.readNet(yolov3_weights, yolov3_cfg)

    # list of files to be removed
    files_to_be_removed = []

    for file_name in tqdm(file_names_from_folder, desc='Detecting pedestrians'):
        # read video
        video_file = os.path.join(video_path, file_name)
        video = cv2.VideoCapture(video_file)

        is_pedestrian_in_video = False
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        pbar = tqdm(total=frame_count, desc='Processing video: {}'.format(file_name), leave=False)

        while video.isOpened():
            # read first frame
            success, frame = video.read()
            # quit if unable to read the video file
            if not success:
                break

            pbar.update(1)

            # create a 4D blob from a frame.
            # this version of net/opencv does not work with batch processing
            blob = cv2.dnn.blobFromImage(
                frame, scale, (416, 416), (0, 0, 0), True, crop=False)

            # set input blob for the network
            net.setInput(blob)

            # run inference through the network
            # and gather predictions from output layers
            outs = net.forward(net.getUnconnectedOutLayersNames())

            # initialization
            class_ids = []
            confidences = []
            boxes = []
            conf_threshold = 0.5
            nms_threshold = 0.4

            # for each detection from each output layer
            # get the confidence, class id, bounding box params
            # and ignore weak detections (confidence < 0.5)
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(detection[0] * frame.shape[1])
                        center_y = int(detection[1] * frame.shape[0])
                        w = int(detection[2] * frame.shape[1])
                        h = int(detection[3] * frame.shape[0])
                        x = center_x - w / 2
                        y = center_y - h / 2
                        class_ids.append(class_id)
                        confidences.append(float(confidence))
                        boxes.append([x, y, w, h])

            indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

            for i in indices:
                if not type(i) == np.int32:
                    i = i[0]
                if class_ids[i] == 0:
                    is_pedestrian_in_video = True
                    break

            if is_pedestrian_in_video:
                break

        if video.isOpened():
            video.release()

        pbar.close()

        if not is_pedestrian_in_video:
            files_to_be_removed.append(file_name)

    print('Number of files to be removed after detection (no pedestrians): {}'.format(
        len(files_to_be_removed)))

    if len(files_to_be_removed):
        logger.debug('Will remove:\n' + ('\n'.join(files_to_be_removed)))

    if remove:
        for file_name in tqdm(files_to_be_removed, desc='Removing the files'):
            os.remove(os.path.join(video_path, file_name))

    logger.info('Finished pedestrian detection.')

    return files_to_be_removed


def has_pedestrian_in_frame(row):
    frame_width = row.get('camera.width', 800)
    frame_height = row.get('camera.height', 600)

    projection_2d = np.array(ast.literal_eval(row['frame.pedestrian.pose.camera'].replace('nan', '"nan"')), dtype=np.float32)
    
    has_pedestrian_in_frame = np.any(
        projection_2d[..., 0] >= 0) & np.any(
        projection_2d[..., 1] >= 0) & np.any(
        projection_2d[..., 0] <= frame_width) & np.any(
        projection_2d[..., 1] <= frame_height)

    return has_pedestrian_in_frame


def trim_useless_rows(csv_path, remove=False, df=None, min_clip_length=30):
    logger.info('Starting CSV rows trimming...')

    df = get_df(csv_path, df)

    original_size = len(df)
    print('Number of rows before trimming: {}'.format(original_size))

    # annotate if frame has even a fragment of a pedestrian
    logger.info('Finding frames with at least a fragment of a pedestrian...')
    has_pedestrian_in_frame_mask = df.apply(has_pedestrian_in_frame, axis=1)

    # get min/max frame number
    logger.info('Calculating min/max frame numbers...')
    grouped = df[has_pedestrian_in_frame_mask].groupby(by=['camera.recording']).aggregate({'frame.idx': ['min', 'max']})
    grouped.columns = grouped.columns.to_flat_index().map(lambda x: '_'.join(x))

    # only keep recordings with at least min_clip_length frames
    lengths = grouped['frame.idx_max'] - grouped['frame.idx_min']
    grouped = grouped[lengths >= min_clip_length]

    # only leave rows between min and max frame number
    # which is not necessary the same as has_pedestrian_in_frame == True
    logger.info('Calculating dataframe mask...')

    df_min_max = df.join(grouped, on='camera.recording', how='left')
    frame_no_mask_min = df_min_max['frame.idx'] >= df_min_max['frame.idx_min']
    frame_no_mask_max = df_min_max['frame.idx'] <= df_min_max['frame.idx_max']
    frame_no_mask = frame_no_mask_min & frame_no_mask_max

    # filter the dataframe
    logger.info('Filtering dataframe...')
    df = df[frame_no_mask]

    new_size = len(df)
    print('Number of rows after trimming: {} (removed {}%).'.format(new_size, round((original_size - new_size) / original_size * 100, 2)))

    if remove and original_size != new_size:
        logger.info('Saving updated CSV.')
        df.to_csv(csv_path, index=False)

    logger.info('Trimming done.')

    return df

def get_df(csv_path, df):
    if df is None:
        logger.info('Loading dataset from CSV (this may take a while)...')
        df = pd.read_csv(csv_path)
        logger.info('Dataset loaded.')
    else:
        logger.info('Using existing dataframe.')
    return df

    