import os
import sys
sys.path.append("./lib/")
import qcsnpe as qc
import glob

from img_helper import _draw_bounding_boxes, preprocess_img, postprocess_img, decode_class_names
from postprocess import filter_boxes, _nms_boxes, sigmoid, post_process_tiny_predictions

import argparse
import numpy as np
import cv2
import time


from vru_helper import update_global_ped_tracking_list
from vru_exact_movement_helper import update_global_ped_tracking_with_movement
from helper.utils.constants import *
import global_var 
from helper.utils.detection_helper import *


from track_obj.centroidtracker import CentroidTracker



vid = True
img = False
multi_thread = True
strides = "32,16"
anchors = "23,27 37,58 81,82 81,82 135,169 344,319"
mask = "3,4,5 0,1,2"
CPU = 0
GPU = 1
DSP = 2
image_size = "416"

model_path = "model_data/tiny_yolov3_model.dlc"
class_name_path = "model_data/coco_classes.txt"
names = decode_class_names(class_name_path)
num_classes = len(decode_class_names(class_name_path))
anchors = np.array(list(map(lambda x: list(map(int, str.split(x, ','))), anchors.split())))
mask = np.array(list(map(lambda x: list(map(int, str.split(x, ','))), mask.split())))
strides = list(map(int, strides.split(',')))
iou_threshold = 0.45
score_threshold = 0.3
max_outputs = 100
in_size = int(image_size)
print(model_path)
out_layers = ["conv2d_12/Conv2D", "conv2d_9/Conv2D"]
model = qc.qcsnpe(model_path,out_layers, CPU)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 10.0, (640, 480))


import cv2
import numpy as np

def user_profiling(frame, ct, frame_cnt, frames):
    img_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = preprocess_img(img_raw, (in_size, in_size))
    img = img.astype(np.float32)
    imgs = img[np.newaxis, ...]
    frame_c = frame.copy()
    print(imgs.shape)
    frame_v = cv2.resize(frame.copy(), (in_size, in_size))
    frame_track = frame_c.copy()
    frame_test = frame_c.copy().astype("uint8")
    frame_track = frame_c.copy().astype("uint8")
    frame_test_vis = np.zeros((frame_c.shape[0], frame_c.shape[1], 3)).astype("uint8")
    frame_test_vis_table = np.zeros((frame_v.shape[0], frame_v.shape[1], 3)).astype("uint8")
    frame_test_inter = frame_c.copy()
    frame_test_all_result = np.zeros((frame_test.shape[0]*2,frame_test.shape[1]*3,3), np.uint8)
    direction_img = np.zeros((frame_c.shape[0], frame_c.shape[1], 3)).astype("uint8")
    event_table_visulizer(frame_v, frame_test_vis_table)
    (H, W) = frame_c.shape[:2] 
    tic = time.time()
    print(imgs.shape)
    output = model.predict(img)
    out1 = output["conv2d_12/BiasAdd:0"]
    out0 = output["conv2d_9/BiasAdd:0"]
    out0=np.reshape(out0,(1,13,13,255))
    out1=np.reshape(out1,(1,26,26,255))
    big_all_boxes, big_all_scores, big_all_classes = post_process_tiny_predictions([out0], anchors, mask, strides, max_outputs, iou_threshold, score_threshold, 0, num_classes)
    small_all_boxes, small_all_scores, small_all_classes = post_process_tiny_predictions([out1], anchors, mask, strides, max_outputs, iou_threshold, score_threshold, 1, num_classes)
    bboxes = np.concatenate([big_all_boxes, small_all_boxes], axis=1)
    scores = np.concatenate([big_all_scores, small_all_scores], axis=1)
    classes = np.concatenate([big_all_classes, small_all_classes], axis=1)
    toc = time.time()
    print((toc - tic)*1000, 'ms', " fps: ", 1/(toc - tic))
    right_boxes, right_classes, right_scores = filter_boxes(bboxes, scores, classes, score_threshold, iou_threshold)
    img, right_boxes = postprocess_img(img, img_raw.shape[1::-1], right_boxes)
    if right_boxes is not None: 
        img = _draw_bounding_boxes(img, right_boxes, right_scores, right_classes, names)
        counter = 0
        face_list = []
        img_size = np.asarray(frame_c.shape)[0:2]
        rects = []
        for bbox, sc, cl in zip(right_boxes,right_scores, right_classes):
            print(cl)
            if cl == 0:
                xmin, ymin, xmax, ymax = (bbox[0]/img.shape[1])*frame_c.shape[1], (bbox[1]/img.shape[0])*frame_c.shape[0], (bbox[2]/img.shape[1])*frame_c.shape[1], (bbox[3]/img.shape[0])*frame_c.shape[0]
                xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                cv2.rectangle(frame_c, (xmin, ymin), (xmax, ymax), (0,0,255), 3)
                box = np.array([xmin, ymin, xmax, ymax])
                rects.append(box.astype("int"))      
        objects = ct.update(rects)
        # loop over the tracked objects
        centroid_pts =[]
        for (objectID, centroid) in objects.items():
            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {}".format(objectID)
            cv2.putText(frame_c, text, (centroid[0] - 10, centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame_c, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
            centroid_pts.append([centroid[0], centroid[1], objectID])
            # discard vanished ped from record
        for id in global_var.global_ped_track_dict.copy():
            rem_id = FLAG_TRUE
            for track in centroid_pts:
                if id == track[2]:
                    rem_id = FLAG_FALSE
                    break
            if rem_id == FLAG_TRUE:
                del global_var.global_ped_track_dict[id]
        frame_test_ped_in_roi_boxes =  update_global_ped_tracking_list(centroid_pts, frame_cnt, frame_test, frame_test_vis, frame_test_inter, direction_img)
        row_cnt = 1
        # movement tracking of ped
        row_cnt = update_global_ped_tracking_with_movement(frame_test_vis, frame_test, frame_test_inter, frame_test_vis_table, direction_img, row_cnt, frame_c)
        firstFrame = False
        frames += 1
        
        cv2.imwrite("frame.jpg",frame_c)
        cv2.imwrite("frame_test_vis.jpg",frame_test_vis)
        cv2.imwrite("frame_test_vis_table.jpg",frame_test_vis_table)
                
        frame_c = cv2.resize(frame_c, (640, 480))
        out.write(frame_c)
    

                
def main():
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--img_folder", default=None, help="image_folder")
    ap.add_argument("-v", "--vid", default=None,help="cam/video_path")
    args = vars(ap.parse_args())

    im_folder_path =  args["img_folder"]
    vid = args["vid"]

    if vid == None and im_folder_path == None:
        print("required command line args atleast ----img_folder <image folder path> or --vid <cam/video_path>")
        exit(0)

    # image inference
    if im_folder_path is not None:
        for image_path in glob.glob(im_folder_path + '/*.jpg'):
            frame = cv2.imread(image_path)
            firstFrame = True
            frames = 0
            detectedPedestrians = {}
            colours = np.random.rand(32, 3)
            ct = CentroidTracker()
            (H, W) = (None, None)
            frame_cnt = 0
            frame_cnt += 1
            print(frame.shape)  
            user_profiling(frame, ct, frame_cnt, frames)
            
            
            
    # video inference
    if vid is not None:
        if vid == "cam":
            video_capture = cv2.VideoCapture(0)
        else:
            video_capture = cv2.VideoCapture(vid)
            
        firstFrame = True
        frames = 0
        detectedPedestrians = {}
        colours = np.random.rand(32, 3)
        ct = CentroidTracker()
        (H, W) = (None, None)
        frame_cnt = 0
        while (video_capture.isOpened()):
            ret, frame = video_capture.read()  # frame shape 640*480*3
            if ret:
                frame_cnt += 1
                user_profiling(frame, ct,frame_cnt, frames)

                if cv2.waitKey(1) & 255==ord('q'):
                    break

        video_capture.release()
        output.release()
        cv2.destroyAllWindows() 
   

if __name__ == '__main__':
    main()
