import urllib.request
import os
import sys
import time

DATA_DIR = "data/%s"
IM_URL = "http://giaothong.hochiminhcity.gov.vn/render/ImageHandler.ashx?id=%s&t=0"
IM_NAME = "im_%s"
SLEEP = 60

def main():
    with open("cam_id", "r") as f:
        cam_ids = [line.split(",")[0] for line in f]

    for id in cam_ids:
        try: os.makedirs(DATA_DIR % id)
        except: pass

    im_counts = {id: len(os.listdir(DATA_DIR % id)) for id in cam_ids}

    while True:
        for id in cam_ids:
            try:
                im = urllib.request.urlopen(IM_URL % id).read()
                with open("%s/%s" % (DATA_DIR % id,
                                     IM_NAME % im_counts[id]),
                          "wb") as im_file:
                    im_file.write(im)
                im_counts[id] += 1
            except:
                print("Failed to collect image", file=sys.stderr)

        time.sleep(SLEEP)

if __name__ == "__main__":
    main()