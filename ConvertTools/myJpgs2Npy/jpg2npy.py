import os
import cv2
import numpy as np

WORK_PATH = "/home/zl/Documents/smartcity/test_set_2K_1_converted"
OUTPUT_PATH = "/home/zl/Documents/smartcity/test_set_2K_1_converted_npy"

PATTERN_FLOW_X = "flow_x_{:05d}.jpg"
PATTERN_FLOW_Y = "flow_y_{:05d}.jpg"

_RESIZE_SIZE = 224


def jpg2npy(work_path, output_path, patternX, patternY):
    packge_list = os.listdir(work_path)
    packge_len = len(packge_list)
    print(packge_len)

    if not os.path.exists(output_path):
        os.mkdir(output_path)
        print("Path:<", output_path, "> is not exist, so create it.")

    record_file = open("npy_list_" + work_path.split("/")[-1] + ".txt", "a")

    index = 0
    for one_video in packge_list:
        index += 1
        video_name = one_video

        if os.path.exists(os.path.join(output_path, video_name + ".npy")):
            # print(video_name, " Already Done!  ", index, "of", packge_len)
            print video_name, " Already Done!  ", index, "of", packge_len
            continue

        video_flow_list = []
        for one in range(1000):
            flowx_name = patternX.format(one + 1)
            flowy_name = patternY.format(one + 1)
            flowx_path = os.path.join(work_path, one_video, flowx_name)
            flowy_path = os.path.join(work_path, one_video, flowy_name)
            if not os.path.exists(flowx_path):
                # print(video_name, "Done!  ", index, "of", packge_len)
                print video_name, "Done!  ", index, "of", packge_len
                break

            oneX = cv2.imread(flowx_path, cv2.IMREAD_GRAYSCALE)
            oneX = cv2.resize(oneX, (_RESIZE_SIZE, _RESIZE_SIZE), interpolation=cv2.INTER_CUBIC)
            oneX = np.reshape(oneX, (_RESIZE_SIZE, _RESIZE_SIZE, 1))

            oneY = cv2.imread(flowy_path, cv2.IMREAD_GRAYSCALE)
            oneY = cv2.resize(oneY, (_RESIZE_SIZE, _RESIZE_SIZE), interpolation=cv2.INTER_CUBIC)
            oneY = np.reshape(oneY, (_RESIZE_SIZE, _RESIZE_SIZE, 1))

            oneCombine = np.concatenate((oneX, oneY), axis=2)

            video_flow_list.append(oneCombine)

        # print(len(video_flow_list), type(video_flow_list))
        video_flow = np.array(video_flow_list)
        # print(len(video_flow), type(video_flow), video_flow.shape)
        video_flow = np.array([video_flow])
        # print(len(video_flow), type(video_flow), video_flow.shape)

        np.save(os.path.join(output_path, video_name+".npy"), video_flow)
        # print("Saved npy shape:", video_flow.shape)
        # print "Saved npy shape:", video_flow.shape
        record_file.write(video_name + "\n")
    record_file.close()


if __name__ == "__main__":
    jpg2npy(work_path=WORK_PATH,
            output_path=OUTPUT_PATH,
            patternX=PATTERN_FLOW_X,
            patternY=PATTERN_FLOW_Y)
