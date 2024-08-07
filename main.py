import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
import matplotlib.pyplot as plt

img = cv2.imread('00.jpg')

cv2.imshow('image',img)
box,lable,c_score = cv.detect_common_objects(img)
output=draw_bbox(box,lable,c_score)
plt.imshow(output)
plt.show()

