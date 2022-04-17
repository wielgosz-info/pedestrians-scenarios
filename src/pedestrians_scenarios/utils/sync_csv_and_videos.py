from enum import unique
import os
import sys
import cv2
import numpy as np
import pandas as pd
import argparse 
import logging

from tqdm import tqdm

logging.getLogger().setLevel(logging.INFO)

def sync_csv_and_videos(args):
    """
    This function makes sure that there are no videos with missing pedestrians (empty videos)
    """
    logging.info('Starting sync csv and videos')

    df = pd.read_csv(args.csv_path)
    file_names_from_folder = os.listdir(args.video_path)
    file_names_from_csv = df['camera.recording'].unique()
    file_names_from_csv = [file_name.split('/')[-1] for file_name in file_names_from_csv]

    # find common elements
    common_elements = set(file_names_from_folder).intersection(file_names_from_csv)
    # subtract common elements from the set of file names from the folder
    elements_to_be_removed = set(file_names_from_folder).difference(common_elements)

    # print common elements 
    logging.info('All the files found: {}'.format(len(file_names_from_folder)))
    logging.info('Number of common files in the folder in the csv:{}'.format(len(common_elements)))
    logging.info('Number of files to be removed from the folder:{}'.format(len(elements_to_be_removed)))

    if args.remove:
        logging.info('Removing the files')
        for element in elements_to_be_removed:
            os.remove(os.path.join(args.video_path, element))
            

# functions
def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def pedestrian_detection_in_video(args):
    """
    Checks if there is a pedesrian visible in the video in at least one frame.
    Taken from:
    https://github.com/arunponnusamy/object-detection-opencv
    """
    logging.info('Starting pedestrian detection in video')

    file_names_from_folder = os.listdir(args.video_path)

    scale = 0.00392
    yolov3_cfg = '/app/src/pedestrians_scenarios/utils/yolov3.cfg'
    yolov3_weights = '/app/src/pedestrians_scenarios/utils/yolov3.weights'

    # check if the files exist if not download them from url and put under current location
    if not os.path.exists(yolov3_cfg):
        logging.info('Downloading yolov3.cfg')
        os.system('wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg -O {}'.format(yolov3_cfg))
    if not os.path.exists(yolov3_weights):
        logging.info('Downloading yolov3.weights')
        os.system('wget https://pjreddie.com/media/files/yolov3.weights -O {}'.format(yolov3_weights))

    # read pre-trained model and config file
    net = cv2.dnn.readNet(yolov3_weights, yolov3_cfg)

    # list of files to be removed
    files_to_be_removed = []

    for file_name in tqdm(file_names_from_folder):
        # read video
        video_path = os.path.join(args.video_path, file_name)
        video = cv2.VideoCapture(video_path)
        # read first frame
        success, frame = video.read()
        # quit if unable to read the video file
        if not success:
            logging.info('Failed to read video')
            sys.exit(1)

        # create a 4D blob from a frame.
        blob = cv2.dnn.blobFromImage(frame, scale, (416, 416), (0, 0, 0), True, crop=False)

        # set input blob for the network
        net.setInput(blob)

        # run inference through the network
        # and gather predictions from output layers
        outs = net.forward(get_output_layers(net))

        # initialization
        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4

        # for each detetion from each output layer 
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
    
        is_pedestrian_in_video = False

        for i in indices:
            i = i[0]
            if class_ids[i] == 0:
                # print('pedestrian is present')
                is_pedestrian_in_video = True
                break

        if not is_pedestrian_in_video:
            files_to_be_removed.append(file_name)

    logging.info('Number of files to be removed after detection (no pedestrians):{}'.format(len(files_to_be_removed)))
    if args.remove:
        for file_name in files_to_be_removed:
            os.remove(os.path.join(args.video_path, file_name))

def clean_csv(args):
    """_summary_
    This function removes the rows with missing values in the csv.
    """
    logging.info('Starting cleaning csv')

    df = pd.read_csv(args.csv_path)
    file_names_from_folder = os.listdir(args.video_path)

    # add 'clips/' to all the elements in the list
    elements_to_keep = ['clips/' + element for element in file_names_from_folder]

    # remove from csv rows which are in the list elements_to_be_removed in the 'camera.recording' column
    number_of_rows_before_removal = len(df)
    df = df[df['camera.recording'].isin(elements_to_keep)]
    number_of_rows_after_removal = len(df)

    # keep in csv rows which are in the list elements_to_be_removed in the 'camera.recording' column
    # df = df[df['camera.recording'].isin(elements_to_be_removed)]
    
    # print number of rows in the csv to be removed
    logging.info('Number of rows to be removed from the csv:{}'.format(number_of_rows_before_removal - number_of_rows_after_removal))

    if args.remove:
        df.to_csv(args.csv_path, index=False)
    

  
 
def main(args, **kwargs):
    """
    Wrapping function for the main function.
    """
    sync_csv_and_videos(args)
    pedestrian_detection_in_video(args)
    clean_csv(args)

if __name__ == '__main__':
    # get command line parameters using parser
    parser = argparse.ArgumentParser("Synch csv and videos")
    parser.add_argument('--csv_path', type=str, default='../../data/pedestrians_scenarios/pedestrians_scenarios.csv',
                        help='Path to csv file')
    parser.add_argument('--video_path', type=str, default='../../data/pedestrians_scenarios/videos',
                        help='Path to videos')  
    parser.add_argument('--remove', action='store_true', default=False,
                        help='Remove files with no pedestrians from the folder') 
    

    args = parser.parse_args()
    main(args)
