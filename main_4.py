import os
import numpy as np
import cv2
from PIL import Image
import gc

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import matplotlib.patches as patches

## import skimage
# from skimage import io

import time
import glob

from sort import Sort

np.random.seed(0)

## colors
#@ 1
# colours = np.random.rand(32, 3)  # used only for display
#@ 2
cmap = plt.get_cmap('tab20b')
# colors = np.array([cmap(i)[:3] for i in np.linspace(0, 1, 20)]) * 255
colors = np.array([cmap(i)[:3] for i in np.linspace(0, 1, 32)]) * 255

def main():

    FILE_NAME = 'test1'

    IMG_PATH = os.path.join('./Data', 'images', f'{FILE_NAME}')
    if not os.path.exists(IMG_PATH):
        os.makedirs(IMG_PATH)

    OUTPUT_PATH = os.path.join('./Output', 'coordinates', f'{FILE_NAME}')
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    FILE_NAME = 'test1'

    IMG_PATH = os.path.join('./Data', 'images', f'{FILE_NAME}')
    if not os.path.exists(IMG_PATH):
        os.makedirs(IMG_PATH)

    OUTPUT_PATH = os.path.join('./Output', 'coordinates', f'{FILE_NAME}')
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
        
    OUTPUT_IMG_PATH = os.path.join('./Output', 'images', f'{FILE_NAME}')
    if not os.path.exists(OUTPUT_IMG_PATH):
        os.makedirs(OUTPUT_IMG_PATH)

    # args = parse_args()

    # display = args.display
    # phase = args.phase

    # total_time = 0.0
    # total_frames = 0

    max_age = 1
    min_hits = 3
    iou_threshold = 0.3

    # Tracker
    mot_tracker = Sort(max_age=max_age,
                       min_hits=min_hits,
                       iou_threshold=iou_threshold)

    # # Matplotlib Interactive
    # plt.ion()
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111, aspect='equal')

    lst = os.listdir(IMG_PATH)  # your directory path
    n_img = len(lst)

    for i in range(1, n_img + 1):
        if i == 101:
            break

        img_path = os.path.join(IMG_PATH, f'frame_{i:04d}.jpg')

        frame = cv2.imread(img_path)
        # frame = skimage.io.imread(img_path)

        if frame is not None:
            frame = frame[..., ::-1]
        else:
            print(f'path: \'{img_path}\' does not have img.')
            exit()

        coords = os.path.join(OUTPUT_PATH, f'frame_{i:04d}.txt')
        dets = np.loadtxt(coords, delimiter=',')

        # dets[:, 2:4] += dets[:, 0:2] # don't need to do this

        ## not using matplot lib
        # ax1.imshow(frame)
        # plt.title(FILE_NAME + ' Tracked Targets')
        
        # start_time = time.time()
        
        trackers = mot_tracker.update(dets)
        
        # cycle_time = time.time() - start_time
        # total_time += cycle_time
        
        ## Needed to avoid error for some reason
        # frame = frame.transpose((1, 2, 0)).astype(np.uint8).copy()
        frame = frame.copy()


        for d in trackers:
            # print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]),file=out_file)
            
            d = d.astype(np.int32)
            
            # cls = classes[int(cls_pred)]
            obj_id = int(d[4]) ## instead of class name we are using class
            conf_score = float(d[5]) ## confidence score, of our detections
            
            
            # color = colors[int(obj_id) % len(colors)]
            color = colors[obj_id % len(colors), :].astype(float)
            
            ## Selecting color
            # 1
            # color = (color * 255.0 / (np.max(color) - np.min(color))).astype(int).tolist()
            
            # 2
            color = colors[int(obj_id) % len(colors)].tolist()
            # color = [i * 255 for i in color]
            
            
            ## Editing the image itself for inserting bounding box
            
            # cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), color, 4)
            cv2.rectangle(img = frame, pt1 = (d[0], d[1]), pt2 = (d[2], d[3]), color = color, thickness = 2)
            
            ## cv2.rectangle(frame, (x1, y1-35), (x1+len(cls)*19+60, y1), color, -1)
            # cv2.rectangle(img = frame, pt1 = (d[0], d[1] - 35), pt2 = (d[0] + len(str(obj_id)) * 19 + 60, d[1]), color = color, thickness = -1)
            
            # frame_text = f'id: {int(obj_id)}'
            frame_text = f'id: {int(obj_id)}, conf: {float(conf_score):.2f}'

            # cv2.putText(frame, cls + "-" + str(int(obj_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
            cv2.putText(img = frame, text = frame_text, org = (d[0], d[1] - 10), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = (255, 255, 255), thickness = 2)
            
            
            ## using Matplot lib to display bounding boxes
            # ax1.add_patch(patches.Rectangle((d[0], d[1]), d[2] - d[0], d[3] - d[1], fill=False, lw=1, ec=colors[d[4] % len(colors), :]))
        
        
        ## not using matplotlib
        # fig.canvas.flush_events()
        # plt.draw()
        # ax1.cla()

        ## saving the image
        output_img_path = f'{OUTPUT_IMG_PATH}/frame_{i:04d}.jpg'
        # print(f'{output_img_path}: {frame.shape}')
        
        ## cv2
        compression_params = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        if not cv2.imwrite(output_img_path, frame, compression_params):
            raise Exception("Could not write image.")
            
        
        ## PIL
        # im = Image.fromarray(frame)
        ## im.save(output_img_path)
        # im.save(output_img_path, format='JPEG',quality=50)
        
        
        del trackers
        del dets
        del coords
        del img_path
        del compression_params
        del output_img_path
        del frame # does it make a difference though ?
        
        gc.collect()


    # plt.close(fig)
    # plt.ioff()

    # del ax1
    # del fig
    # gc.collect()


if __name__ == '__main__':
    main()
