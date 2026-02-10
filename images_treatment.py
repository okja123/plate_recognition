import itertools
import math
import kagglehub
import os 
import cv2
import numpy as np
import glob

#pip install kagglehub opencv-python numpy
def download_licence_plate():
    os.environ['KAGGLEHUB_CACHE'] = './'
    path = kagglehub.dataset_download("abdelhamidzakaria/european-license-plates-dataset")
    print("path ", path)

PATH_TO_DATASET = "./datasets/abdelhamidzakaria/european-license-plates-dataset/versions/1/dataset_final"
files = [f for f in glob.glob(PATH_TO_DATASET+"/test/*.png")]
#img = cv2.imread(files[4],cv2.IMREAD_GRAYSCALE)
# 2marche
img = cv2.imread(files[4],cv2.IMREAD_GRAYSCALE)
cv2.imshow("image", img)

_ ,img_treshold = cv2.threshold(img,int(np.mean(img)),255,cv2.THRESH_BINARY)

black_img = np.zeros(img.shape)
black_img_corner = np.zeros(img.shape)
contours, _ = cv2.findContours(img_treshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
plaque = max(contours, key=cv2.contourArea)
cv2.polylines(black_img_corner,plaque,1,(255,255,255),1 )


l = len(plaque)
vector2_zero = np.zeros(2)
corner = []
shape = img.shape
for i in range(l):
    v1  = np.subtract(plaque[i],plaque[i-1])
    v2  = np.subtract(plaque[i],plaque[(i+1)%l])
    if np.sum(np.abs(v1)) == np.sum(np.abs(v2))  == 1 and not np.array_equal(np.add(v1,v2)[0],vector2_zero):
        corner.append([plaque[i][0][0],plaque[i][0][1]])
        black_img[plaque[i][0][1]][plaque[i][0][0]] = 255
        
corner = np.array(corner).astype(np.float32)

cv2.imshow("image contpoyr", black_img_corner)
cv2.imshow("image corner", black_img)

def transform():
    a = np.array([
        [50,45],
        [420,17],
        [55,122],
        [420,126],
    ],dtype=np.float32)

    b = np.array([
        [0,0],
        [shape[1],0],
        [0,shape[0]],
        [shape[1],shape[0]],
    ],dtype=np.float32)

    print(a)
    print(corner)
    c = np.array([corner[0],corner[3],corner[1],corner[2]])

    T = cv2.getPerspectiveTransform(c ,b)
    img_trans = cv2.warpPerspective(img,T,(shape[1],shape[0]))
    cv2.imshow("image contpoyra", img_trans)



#cv2.imwrite("test.png",black_img)
cv2.waitKey(0)
cv2.destroyAllWindows()