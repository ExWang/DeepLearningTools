import os
import os.path as osp
import json
import numpy as np
import configparser
import cv2

import myJsonHelper as mjh
import myMatHelper as mmh

TARGET_PATH = "examples/res/alphapose-results.json"
TARGET_DIR = "examples/res"
OUTPUT_DIR = "examples/MOT16_fin"
CONFIG_DIR = "/media/sbaer/Warehouse/Dataset/Tracking/MOT16/train"
CONFIG_NAME = "seqinfo.ini"

EXPD_RATE = 0.0  # The b-box needs to surround the person, and leave some space.
THRESH_SCORE = 0.8

COLOR_LIST = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255),
              (0, 0, 125), (0, 125, 0), (125, 0, 0), (125, 125, 0), (125, 0, 125), (0, 125, 125)]


def getBond(pos_list, shape):
    x1, y1 = np.nanmin(pos_list, axis=0)
    x2, y2 = np.nanmax(pos_list, axis=0)

    height = abs(y2 - y1)
    width = abs(x2 - x1)

    expd_x = width * EXPD_RATE / 2.0
    expd_y = height * EXPD_RATE / 2.0

    expd_width = width * (1 + EXPD_RATE)
    expd_height = height * (1 + EXPD_RATE)

    x1 = max(0, x1 - expd_x)
    y1 = max(0, y1 - expd_y)

    # x2 = min(shape[0], x2 + expd_x)
    # y2 = min(shape[1], y2 + expd_y)

    x1 = round(x1, 3)
    y1 = round(y1, 3)
    # x2 = round(x2, 3)
    # y2 = round(y2, 3)
    expd_width = round(expd_width, 3)
    expd_height = round(expd_height, 3)

    left = x1
    top = y1

    # return [x1, y1, x2, y2]
    return [left, top, expd_width, expd_height]


def getBond_v2(pos_list, shape):
    x1, y1 = np.nanmin(pos_list, axis=0)
    x2, y2 = np.nanmax(pos_list, axis=0)

    # print(x1, y1, x2, y2)

    height = abs(y2 - y1)
    width = abs(x2 - x1)

    expd_x = width * EXPD_RATE / 2.0
    expd_y = height * EXPD_RATE / 2.0

    x1 = max(0, x1 - expd_x)
    y1 = max(0, y1 - expd_y)

    x2 = min(shape[0], x2 + expd_x)
    y2 = min(shape[1], y2 + expd_y)

    # print(x1, y1, x2, y2)

    expd_width = abs(x2 - x1)
    expd_height = abs(y2 - y1)

    x1 = round(x1, 3)
    y1 = round(y1, 3)
    expd_width = round(expd_width, 3)
    expd_height = round(expd_height, 3)

    left = x1
    top = y1

    return [left, top, expd_width, expd_height]


def getBond_fake(pos_list, shape):
    max_bound = np.nanmax(pos_list, axis=0)
    min_bound = np.nanmin(pos_list, axis=0)

    [x1, y1], _ = mmh.findNearestPoint(min_bound)
    [x2, y2], _ = mmh.findNearestPoint(max_bound)

    height = abs(y2 - y1)
    width = abs(x2 - x1)

    expd_x = int(round(width * EXPD_RATE))
    expd_y = int(round(height * EXPD_RATE))

    x1 = max(0, x1 - expd_x)
    y1 = max(0, y1 - expd_y)

    x2 = min(shape[0], x2 + expd_x)
    y2 = min(shape[1], y2 + expd_y)

    return [x1, y1, x2, y2]


def add_one(src, add):
    assert type(src) is str
    if len(src) == 0:
        return str(add)
    else:
        return src + ',' + str(add)


def tran2MOT16(src, cfg):
    print("> Transfer target to MOT-16 form <")
    print("<CONFIG>", cfg)
    dst_list = []
    idx = 1
    for oneFrame in src:
        code = 1
        for onePer in oneFrame:
            # print(onePer)
            score = onePer['score']
            keyps = onePer['keypoints']

            if score < THRESH_SCORE:
                continue
            # print(score)
            # print(keyps)
            keyps = np.array(keyps)
            keyps = np.reshape(keyps, (-1, 3))
            keyscores = keyps.copy()
            keyscores = np.delete(keyscores, (0, 1), axis=1)
            keyps = np.delete(keyps, 2, axis=1)
            # bbox = getBond(keyps, [cfg['w'], cfg['h']])
            bbox = getBond_v2(keyps, [cfg['w'], cfg['h']])
            # bbox_2 = getBond_v2(keyps, [cfg['w'], cfg['h']])

            # print(bbox, "<====>", bbox_2)

            frame = idx
            id = code
            # id = -1
            left = bbox[0]
            top = bbox[1]
            width = bbox[2]
            height = bbox[3]
            conf = -1
            dx = -1
            dy = -1
            dz = -1

            dst_one = ""
            dst_one = add_one(dst_one, frame)
            dst_one = add_one(dst_one, id)
            dst_one = add_one(dst_one, left)
            dst_one = add_one(dst_one, top)
            dst_one = add_one(dst_one, width)
            dst_one = add_one(dst_one, height)
            dst_one = add_one(dst_one, conf)
            dst_one = add_one(dst_one, dx)
            dst_one = add_one(dst_one, dy)
            dst_one = add_one(dst_one, dz)
            dst_one += "\n"

            dst_list.append(dst_one)
            code += 1

        idx += 1

    return dst_list


def tran2MOT16_fake(src, cfg):
    print("> Transfer target to MOT-16 form <")
    print("<CONFIG>", cfg)
    dst_list = []
    idx = 1
    for oneFrame in src:
        code = 1
        dst_one_list = []
        for onePer in oneFrame:
            # print(onePer)
            score = onePer['score']
            keyps = onePer['keypoints']

            if score < THRESH_SCORE:
                continue
            # print(score)
            # print(keyps)
            keyps = np.array(keyps)
            keyps = np.reshape(keyps, (-1, 3))
            keyscores = keyps.copy()
            keyscores = np.delete(keyscores, (0, 1), axis=1)
            keyps = np.delete(keyps, 2, axis=1)
            bbox = getBond_fake(keyps, [cfg['w'], cfg['h']])

            id = code
            dst_one = {'id': id, 'box': bbox}
            dst_one_list.append(dst_one)
            code += 1

        idx += 1
        dst_list.append(dst_one_list)

    return dst_list


def readAlphaRes(path_json):
    print("<READ> ", path_json)

    data_json = {}
    with open(path_json) as jf:
        data_json = json.load(jf)
    num_frame = 0
    for oneData in data_json:
        oneName = oneData['image_id']
        if num_frame < int(oneName.split(".")[0]):
            num_frame = int(oneName.split(".")[0])
    data_frame = [[] for i in range(num_frame)]

    for oneData in data_json:
        oneName = oneData['image_id']
        oneFrameCode = int(oneName.split(".")[0]) - 1
        oneKeypoints = oneData['keypoints']
        oneScore = oneData['score']
        data_frame[oneFrameCode].append({'keypoints': oneKeypoints, 'score': oneScore})
    print("<SUCCESS>")
    return data_frame


def readConfig(name, dir_cfg):
    print(name)
    print(dir_cfg)
    config = configparser.ConfigParser()
    dir_target = osp.join(dir_cfg, name)
    path_cfg = ""
    if CONFIG_NAME in os.listdir(dir_target):
        path_cfg = osp.join(dir_target, CONFIG_NAME)
    else:
        raise FileNotFoundError
    config.read(path_cfg)
    imWidth = config.get("Sequence", "imWidth")
    imHeight = config.get("Sequence", "imHeight")
    return {'w': int(imWidth), 'h': int(imHeight)}


def writeData(data, dst_path):
    print("<WRITE>", dst_path)
    with open(dst_path, "w") as wf:
        wf.writelines(data)
    return 1


def myWorker_1(dir_target, dir_output, dir_cfg):
    print("<START> Work in:", dir_target)
    dirs = os.listdir(dir_target)
    dirs.sort()
    if not osp.exists(dir_output):
        os.mkdir(dir_output)
    for oneDir in dirs:
        targetDir = osp.join(dir_target, oneDir)
        targetJson = os.listdir(targetDir)[0]
        targetRelativePath = osp.join(targetDir, targetJson)
        targetName = targetJson.split(".")[0]
        print("<NOW>", targetName)
        targetData = readAlphaRes(targetRelativePath)
        cfg = readConfig(targetName, dir_cfg)
        outData = tran2MOT16(targetData, cfg)
        outFilePath = osp.join(dir_output, targetName + ".txt")
        #
        # print(outData)
        writeData(data=outData, dst_path=outFilePath)

        # exit(0)
    return 1


def myWorker_2(dir_target, dir_output, dir_cfg):
    print("<START> Work in:", dir_target)
    dirs = os.listdir(dir_target)
    dirs.sort()
    if not osp.exists(dir_output):
        os.mkdir(dir_output)
    for oneDir in dirs:
        targetDir = osp.join(dir_target, oneDir)
        targetJson = os.listdir(targetDir)[0]
        targetRelativePath = osp.join(targetDir, targetJson)
        targetName = targetJson.split(".")[0]
        print("<NOW>", targetName)
        targetData = readAlphaRes(targetRelativePath)
        cfg = readConfig(targetName, dir_cfg)
        outData = tran2MOT16_fake(targetData, cfg)

        dataset_dir = osp.join(dir_cfg, targetName)
        video_dir = osp.join(dataset_dir, "img1")
        picList = os.listdir(video_dir)
        picList.sort()  # let list be ordered
        frame_num = len(picList)

        cv2.namedWindow('demo', cv2.WINDOW_AUTOSIZE)
        cv2.startWindowThread()

        i = 0
        for pic in picList:
            img_path = os.path.join(video_dir, pic)
            img_raw = cv2.imread(img_path)
            img = img_raw.copy()

            now_frame = outData[i]

            for one in now_frame:
                one_id = one['id']
                box = one['box']
                mmh.drawRect(img, box[0], box[1], box[2], box[3],
                             color=COLOR_LIST[one_id % len(COLOR_LIST)])
                mjh.drawNum(img, one_id, box[0], box[1],
                            color=COLOR_LIST[one_id % len(COLOR_LIST)])

            cv2.imshow('demo', img)
            while True:
                key = cv2.waitKey()
                if key == 83 or key == 84 or key == 27:
                    break
            if key == 27:
                break
            i += 1

        cv2.destroyAllWindows()

        # outFilePath = osp.join(dir_output, targetName + ".txt")

        # writeData(data=outData, dst_path=outFilePath)

        exit(0)
    return 1


def myTester_1(path_json):
    print("Do something.")
    data_json = {}
    with open(path_json) as jf:
        data_json = json.load(jf)
    print(type(data_json), len(data_json))
    for oneData in data_json:
        oneName = oneData['image_id']
        print(oneName)
    oneData = data_json[0]
    print(oneData.keys())
    oneKeys = oneData['keypoints']
    print(oneKeys)
    print(len(oneKeys), len(oneKeys) / 3)


if __name__ == "__main__":
    # readAlphaRes(TARGET_PATH)
    myWorker_1(TARGET_DIR, OUTPUT_DIR, CONFIG_DIR)
    print("Done!")
