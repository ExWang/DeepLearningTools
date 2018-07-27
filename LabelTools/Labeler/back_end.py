from PythonAPI.pycocotools.coco import COCO
import numpy as np

dataDir = '.'
dataType = 'train2014'
annFile = '{}/annotations/captions_{}.json'.format(dataDir, dataType)
coco_caps = COCO(annFile)


def getCaption(img_name):
    print("Get img:", img_name)
    IMG_ID = int(img_name.split("_")[-1].split(".")[0])
    imgIds = coco_caps.getImgIds(imgIds=[IMG_ID])
    img = coco_caps.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]
    annIds = coco_caps.getAnnIds(imgIds=img['id'])
    anns = coco_caps.loadAnns(annIds)
    captions_list = coco_caps.showAnns(anns)
    str_caption = ""
    i = 1
    for oneSen in captions_list:
        str_caption += str(i)
        str_caption += ". "
        str_caption += oneSen
        str_caption += '\n'
        i += 1

    return str_caption[:-1]


if __name__ == "__main__":
    ret = getCaption("COCO_train2014_000000000009.jpg")
    print(ret)
