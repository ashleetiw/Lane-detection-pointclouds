#!/usr/bin/env python3
import cv2
import numpy as np
import os
from os.path import isfile, join



def convert_frames_to_video(pathIn,pathOut,fps):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
    #for sorting the file names properly
    files.sort()

    # i1=files.index('um_000006.png')
    # i2=files.index('um_000008.png')
    # # print(i1)
    # files=files[i1:i2+1]

    # print(files)
    for i in range(len(files)):
        filename=pathIn + files[i]
        # print(filename)
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        print(filename)
        #inserting the frames into an image array
        frame_array.append(img)

    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()


def main():
    pathIn= '/home/ashlee/person_project/lidar/winter_project/video/'
    pathOut = 'reatime_video.avi'
    fps = 8.0
    convert_frames_to_video(pathIn, pathOut, fps)


if __name__=="__main__":
    main()