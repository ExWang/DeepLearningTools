import cv2
import os
import random
import numpy as np
import time

PATH_VIDEOS = '/media/sbaer/Warehouse/Dataset/smartcity_action90_zip/test_set'

PATH_FRAMES = '/media/sbaer/Entertainment/dataset/test_set_converted/fake'

_SPACE = " "

CLASS_NAME_FILE = '/media/sbaer/Warehouse/Dataset/smartcity_action90_zip/class_name.txt'

OUT_MAIN_PATH = '/media/sbaer/Entertainment/dataset/test_set_converted/smart90/'
OUT_TOTAL_FILE_PATH = OUT_MAIN_PATH + 'rgb.txt'
OUT_LIST_FILE_PATH = OUT_MAIN_PATH + 'testlist01.txt'


# TVL1 = cv2.DualTVL1OpticalFlow_create()

def get_reverse_Class_dict(file_path):
    print("Get class dict from:", file_path)
    with open(file_path, "r") as txtFile:
        lines = txtFile.readlines()
        print("Class num:", len(lines))
    ret = {}
    index = 0
    for one in lines:
        ret[index] = one.strip("\n")
        index += 1
    return ret


def compute_TVL1(prev, curr, bound=15):
    """Compute the TV-L1 optical flow."""
    TVL1 = cv2.DualTVL1OpticalFlow_create()
    flow = TVL1.calc(prev, curr, None)
    assert flow.dtype == np.float32

    flow = (flow + bound) * (255.0 / (2 * bound))
    flow = np.round(flow).astype(int)
    flow[flow >= 255] = 255
    flow[flow <= 0] = 0

    return flow


def compute_one_video(video_path):
    cap = cv2.VideoCapture(video_path)
    # print videoFullName
    flag_first = True
    prev = ""
    flow = []
    now_time = time.time()
    count = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_CUBIC)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if flag_first:
            prev = frame
            flag_first = False
            continue
        curr = frame
        tmp_flow = compute_TVL1(prev=prev, curr=curr)
        print("Frame:", count, "Shape:",tmp_flow.shape)
        flow.append(tmp_flow)
        prev = curr
        count += 1
    print("Cost:", time.time() - now_time)
    cap.release()
    frames = flow
    return frames


def work():
    path_videos = PATH_VIDEOS
    path_frames = PATH_FRAMES

    class_dict = get_reverse_Class_dict(CLASS_NAME_FILE)

    if not os.path.exists(path_frames):
        os.makedirs(path_frames)

    if not os.path.exists(OUT_MAIN_PATH):
        os.makedirs(OUT_MAIN_PATH)

    videos = os.listdir(path_videos)
    print(len(videos))
    videos.sort()

    # f_total = open(OUT_TOTAL_FILE_PATH, "a")
    # f_list = open(OUT_LIST_FILE_PATH, "a")

    for i, videoName in enumerate(videos):
        videoDirName = videoName.split('.')[0]
        path_save = os.path.join(path_frames, videoDirName)
        if not os.path.exists(path_save):
            os.makedirs(path_save)
        else:
            print("{:s} has already completed before. ({:d}/{:d})".format(videoName, i + 1, len(videos)))
            continue
        videoFullName = os.path.join(path_videos, videoName)

        video_flow = compute_one_video(videoFullName)

        # # Something write to files
        # c1 = videoDirName
        # c2 = path_save
        # c3 = count
        # c4 = random.randint(0, 89)
        # total = c1 + _SPACE + c2 + _SPACE + str(c3) + _SPACE + str(c4) + '\n'
        # # print(total)
        # class_name = class_dict[c4]
        # other_total = class_name + '/' + c1 + '.avi\n'
        # # print(other_total)
        # f_total.write(total)
        # f_list.write(other_total)
    return video_flow

    # f_total.close()
    # f_list.close()


if __name__ == "__main__":
    work()
