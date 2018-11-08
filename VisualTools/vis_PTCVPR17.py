import scipy.io as sio
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import math

WORK_DIR = "/home/sbaer/workspace/Sample/PoseTrack-CVPR2017/data/bonn-multiperson-posetrack"
VIDEO_DIR = WORK_DIR + "/videos"

ANNO_MAT = WORK_DIR + "/annolist/test/annolist.mat"
PT_MUL_PRED = WORK_DIR + "/pt-multicut/prediction_3_060470979.mat"

THRESH_MinAppTimes = 7
EXPD_BOUND = 10  # The b-box needs to surround the person, and leave some space.

COLOR_LIST = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255),
              (0, 0, 125), (0, 125, 0), (125, 0, 0), (125, 125, 0), (125, 0, 125), (0, 125, 125)]


def getValNF(target):
    return str(target[0][0])


def getValN(target):
    return int(target[0][0])


def showPLT(target):
    img = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()


def findNearestPoint(onePoint):
    high = np.ceil(onePoint)
    low = np.floor(onePoint)
    p1 = np.array([high[0], low[1]])
    p2 = np.array([low[0], high[1]])
    truePot = high
    trueDist = np.linalg.norm(onePoint - truePot)
    for pot in [low, p1, p2]:
        tmpDist = np.linalg.norm(onePoint - pot)
        if tmpDist < trueDist:
            truePot = pot
            trueDist = tmpDist
    truePot = truePot.astype(np.int32)
    return truePot, trueDist


def drawDot(img, x, y, color=(0, 0, 255), raidus=4):
    return cv2.circle(img, (x, y), raidus, color, -1)


def drawRect(img, x1, y1, x2, y2, color=(0, 0, 255), bnd=2):
    return cv2.rectangle(img, (x1, y1), (x2, y2), color, bnd)


def drawBond(img, pos_list, index):
    max_bound = np.nanmax(pos_list, axis=0)
    min_bound = np.nanmin(pos_list, axis=0)
    [x1, y1], _ = findNearestPoint(min_bound)
    [x2, y2], _ = findNearestPoint(max_bound)
    expd = EXPD_BOUND
    x1 = max(0, x1 - expd)
    y1 = max(0, y1 - expd)

    x2 = min(img.shape[1], x2 + expd)
    y2 = min(img.shape[0], y2 + expd)
    # print("=======", index, "========")
    # print(img.shape[1], img.shape[0])
    # print(x1, y1)
    # print(x2, y2)
    return drawRect(img, x1, y1, x2, y2, color=COLOR_LIST[index])


def matRead(file_path):
    print("Read MATLAB file:", file_path)
    mat_file = sio.matlab.loadmat(file_path)
    mat_filename = file_path.split("/")[-1].split(".")[0]

    if mat_filename == "annolist":
        res = {}
        print("===> LOAD: annolist.")
        print(mat_file.keys())
        data = mat_file[mat_filename]
        print(data.shape)
        row_num = data.shape[0]
        col_num_frames = data['num_frames']
        col_name = data['name']
        col_num_persons = data['num_persons']
        col_annopoints = data['annopoints']
        list_name = []
        list_num_frames = []
        list_num_persons = []
        list_annopoints = []
        for idx in range(row_num):
            list_name.append(getValN(col_name[idx]))
            list_num_frames.append(getValNF(col_num_frames[idx]))
            list_num_persons.append(getValNF(col_num_persons[idx]))
            list_annopoints.append(col_annopoints[idx])

        res['row_num'] = row_num
        res['list_name'] = list_name
        res['list_num_frames'] = list_num_frames
        res['list_num_persons'] = list_num_persons
        res['list_annopoints'] = list_annopoints

    elif mat_filename.split("_")[0] == "prediction":
        res = []
        print("===> LOAD: pt-multicut prediction")
        print(mat_file.keys())
        data = mat_file['people']
        print(data.shape)
        num_frames = data.shape[1]
        num_targets = data.shape[0]
        for idx in range(num_targets):
            one_target = data[idx]
            one_num_appr = 0
            one_list_appr = [0 for _ in range(num_frames)]
            for idx_fram in range(num_frames):
                one = one_target[idx_fram]
                if len(one) > 1:
                    one_num_appr += 1
                    one_list_appr[idx_fram] = 1

            one_target_dict = {'appear_times': one_num_appr,
                               'appear_list': one_list_appr,
                               'data': one_target}
            res.append(one_target_dict)
        print("Result length:", len(res))

    else:
        print("ERROR: Load unrecognized format mat.")
        raise NameError

    print(">======= Finished Loading =======<")

    return res


def myWorker(anno_path, ptPred_path, video_dir):
    print("===== Do something =====")
    anno_info = matRead(anno_path)
    print(anno_info.keys())
    pred_data = matRead(ptPred_path)

    cv2.namedWindow('demo', cv2.WINDOW_AUTOSIZE)

    video_name_list = os.listdir(video_dir)
    print(video_name_list)
    for idx in range(len(video_name_list)):
        video_pic_dir = os.path.join(video_dir, video_name_list[idx])
        picList = os.listdir(video_pic_dir)
        picList.sort()  # let list be ordered
        print(picList)

        i = 0
        for pic in picList:
            print("Frame ===>", i+1)
            img_path = os.path.join(video_pic_dir, pic)
            img_raw = cv2.imread(img_path)
            img_h, img_w, img_c = img_raw.shape
            img = img_raw

            code = 0
            for onePerson in pred_data:
                if onePerson['appear_times'] < THRESH_MinAppTimes:
                    continue
                if onePerson['appear_list'][i] == 0:
                    continue
                jointPos = onePerson['data'][i]
                for onePoint in jointPos:
                    if not np.isnan(onePoint).any():  # not nan point
                        pot, _ = findNearestPoint(onePoint)
                        drawDot(img, pot[0], pot[1])
                drawBond(img, jointPos, code)
                # exit(0)
                code += 1
            # exit(0)

            # =====================IMG SHOW================
            cv2.imshow('demo', img)
            while True:
                key = cv2.waitKey()
                if key == 83 or key == 84 or key == 27:
                    break
            if key == 27:
                break
            i += 1

        cv2.destroyAllWindows()


if __name__ == "__main__":
    myWorker(ANNO_MAT, PT_MUL_PRED, VIDEO_DIR)
