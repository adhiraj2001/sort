import time
import os
import glob

import numpy as np
import cv2
import gc

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# from skimage import io
import skimage

from sort import Sort

np.random.seed(0)

colours = np.random.rand(32, 3)  # used only for display


def main():

    FILE_NAME = 'test1'

    IMG_PATH = os.path.join('./Data', 'images', f'{FILE_NAME}')
    if not os.path.exists(IMG_PATH):
        os.makedirs(IMG_PATH)

    OUTPUT_PATH = os.path.join('./Output', 'coordinates', f'{FILE_NAME}')
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    # args = parse_args()

    # display = args.display
    # phase = args.phase

    # total_time = 0.0
    # total_frames = 0

    max_age = 1
    min_hits = 3
    iou_threshold = 0.01

    # Tracker
    mot_tracker = Sort(max_age=max_age,
                       min_hits=min_hits,
                       iou_threshold=iou_threshold)

    # Matplotlib Interactive
    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(111, aspect='equal')

    lst = os.listdir(IMG_PATH)  # your directory path
    n_img = len(lst)

    for i in range(1, n_img + 1):
        if i == 101:
            break

        img_path = os.path.join(IMG_PATH, f'frame_{i:04d}.jpg')
        # img = cv2.imread(img_path)
        img = skimage.io.imread(img_path)

        if img is not None:
            img = img[..., ::-1]
        else:
            print(f'path: \'{img_path}\' does not have img.')
            exit()

        frame = os.path.join(OUTPUT_PATH, f'frame_{i:04d}.txt')
        dets = np.loadtxt(frame, delimiter=',')

        # dets[:, 2:4] += dets[:, 0:2] # don't need to do this

        ax1.imshow(img)
        plt.title(FILE_NAME + ' Tracked Targets')

        # start_time = time.time()

        trackers = mot_tracker.update(dets)

        # cycle_time = time.time() - start_time
        # total_time += cycle_time

        for d in trackers:
            # print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]),file=out_file)

            d = d.astype(np.int32)
            ax1.add_patch(patches.Rectangle(
                (d[0], d[1]), d[2] - d[0], d[3] - d[1], fill=False, lw=1, ec=colours[d[4] % 32, :]))

        fig.canvas.flush_events()
        plt.draw()
        ax1.cla()

        del trackers
        del dets
        del frame
        del img_path
        del img  # does it make a difference though ?
        gc.collect()

    plt.close(fig)
    plt.ioff()

    del ax1
    del fig
    gc.collect()


if __name__ == '__main__':
    main()
