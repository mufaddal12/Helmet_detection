from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
from darknet import darknet
from sys import argv


def convertBack(x, y, w, h, inputshape, imgshape):
    inx, iny = inputshape
    imgy, imgx, _ = imgshape

    x = (float(x / float(inx)) * imgx)
    w = (float(w / float(inx)) * imgx)
    y = (float(y / float(iny)) * imgy)
    h = (float(h / float(iny)) * imgy)

    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    print(xmin, ymin, xmax, ymax)
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img, inputshape):

    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]

        xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w),
                                             float(h), inputshape, img.shape)
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        cv2.putText(img, detection[0].decode(), (pt1[0], pt1[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 0], 1)
    return img


netMain = None
metaMain = None
altNames = None


def YOLO():

    global metaMain, netMain, altNames
    configPath = "Configuration/yolov4_obj.cfg"
    weightPath = "Weights/yolov4_obj_final.weights"
    metaPath = "Configuration/detector.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath) + "`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath) + "`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath) + "`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode("ascii"),
                                          weightPath.encode("ascii"), 0,
                                          1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    #cap = cv2.VideoCapture(0)
    imagepath = argv[1]
    if os.path.exists(imagepath):
        name = imagepath.split('/')[-1].split('.')[0]
        frame_read = cv2.imread(imagepath)
    else:
        print("Incorrect path to image")
        return
    print("Starting the YOLO loop...")

    inputshape = (darknet.network_width(netMain),
                  darknet.network_height(netMain))

    darknet_image = darknet.make_image(inputshape[0], inputshape[1], 3)
    prev_time = time.time()
    frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb,
                               inputshape,
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())

    detections = darknet.detect_image(netMain,
                                      metaMain,
                                      darknet_image,
                                      thresh=0.2)
    image = cvDrawBoxes(detections, frame_rgb, inputshape)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print("FPS : ", end='')
    print(1.0 / (time.time() - prev_time))
    cv2.imwrite(name + "_output.jpg", image)


if __name__ == "__main__":
    YOLO()
