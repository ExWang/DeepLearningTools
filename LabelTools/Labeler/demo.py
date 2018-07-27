from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
from PIL import Image

pylab.rcParams['figure.figsize'] = (8.0, 10.0)

print('Start')
dataDir = '..'
dataType = 'val2014'
annFile = '{}/annotations/captions_{}.json'.format(dataDir, dataType)

print(annFile)

coco_caps = COCO(annFile)

IMG_ID = 115006

# catIds = coco_caps.getCatIds(catNms=['person','dog','skateboard']);
# imgIds = coco_caps.getImgIds(catIds=catIds );
imgIds = coco_caps.getImgIds(imgIds=[IMG_ID])
img = coco_caps.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]
img_name = img['file_name']
img_path = '/home/wsh/coco/val2014/' + img_name
image = Image.open(img_path)

res = None
annIds = coco_caps.getAnnIds(imgIds=img['id'])
anns = coco_caps.loadAnns(annIds)
print('---------------------------', IMG_ID, '--------------------------------')
coco_caps.showAnns(anns)
image.show()
print('----------------------------------------------------------------')

# plt.imshow(image); plt.axis('off'); plt.show()

