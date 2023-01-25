# import onnxruntime as ort
# import numpy as np
# import cv2

# ort_sess = ort.InferenceSession('./best.onnx')

# #get img
# img = cv2.imread('./img.jpg')

# img = cv2.resize(img, (544, 960))

# img = cv2.normalize(img, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

# #get all channels
# r = img[:,:,0]
# g = img[:,:,1]
# b = img[:,:,2]

# #merge channels
# img = [r,g,b]

# img = np.array(img)

# print(img.shape)

# img = [img]

# img = np.array(img)

# print(img.shape)

# label_name = ort_sess.get_outputs()[0].name

# outputs = ort_sess.run([label_name], {ort_sess.get_inputs()[0].name : img})[0]
# # Print Result
# print(outputs.shape)
# prediction = np.array(outputs).squeeze()
# prediction=np.argmax(prediction, axis=0)
# print(prediction.shape)





import cv2
import time
import requests
import random
import numpy as np
import onnxruntime as ort
from PIL import Image
from pathlib import Path
from collections import OrderedDict,namedtuple

w = "./bestv2.onnx"
# img = cv2.imread('./img.jpg')
vid = cv2.VideoCapture(0)

providers = ['CPUExecutionProvider']
session = ort.InferenceSession(w, providers=providers)


def letterbox(im, new_shape=(960, 544), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)

names = ['1','2','3','4']
colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(names)}


while True:
    ok, img = vid.read()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    image = img.copy()
    # image, ratio, dwdh = letterbox(image, auto=False)
    ratio = 9/16
    dwdh = (540, 960)
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, 0)
    image = np.ascontiguousarray(image)

    im = image.astype(np.float32)
    im /= 255
    im.shape

    outname = [i.name for i in session.get_outputs()]
    outname

    inname = [i.name for i in session.get_inputs()]
    inname

    inp = {inname[0]:im}

    outputs = session.run(outname, inp)[0]

    ori_images = [img.copy()]

    for i,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(outputs):
        image = ori_images[int(batch_id)]
        box = np.array([x0,y0,x1,y1])
        box -= np.array(dwdh*2)
        box /= ratio
        box = box.round().astype(np.int32).tolist()
        cls_id = int(cls_id)
        score = round(float(score),3)
        name = names[cls_id]
        color = colors[name]
        name += ' '+str(score)
        cv2.rectangle(image,box[:2],box[2:],color,2)
        cv2.putText(image,name,(box[0], box[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,[225, 255, 255],thickness=2)  

    img = cv2.cvtColor(ori_images[0], cv2.COLOR_RGB2BGR)
    cv2.imshow('img',img)
    k = cv2.waitKey(1)
    if k == 27:
        break

cv2.destroyAllWindows()