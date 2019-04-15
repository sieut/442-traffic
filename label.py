import cv2
import os

def add_labels(cam_id, start_id, end_id, label):
    with open("data/labels", "a") as out:
        for i in range(start_id, end_id + 1):
            out.write("%s,%s\n" % (key(cam_id, i), label))


def key(cam_id, im_id):
    return "%s_%s" % (cam_id, im_id)