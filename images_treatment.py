from scipy.spatial import cKDTree
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

images = [cv2.imread(f,cv2.IMREAD_GRAYSCALE) for f in glob.glob(PATH_TO_DATASET+"/test/*.png")]
image = images[1]

def image_treatment(img):
    _ ,img_treshold = cv2.threshold(img,int(np.mean(img)),255,cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(img_treshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours_plaque = max(contours, key=cv2.contourArea)
    return(img_treshold,contours,contours_plaque)

def detection_corner1(contours):
    l = len(contours)
    vector2_zero = np.zeros(2)
    corner = []
    for i in range(l):
        v1  = np.subtract(contours[i],contours[i-1])
        v2  = np.subtract(contours[i],contours[(i+1)%l])
        if np.sum(np.abs(v1)) == np.sum(np.abs(v2))  == 1 and not np.array_equal(np.add(v1,v2)[0],vector2_zero):
            corner.append(contours[i])
    return corner

def detection_corner2(contours_input,range_point):
    contours = np.squeeze(contours_input, axis=1)
    l = len(contours)
    scores = np.zeros(l,dtype=np.int16)
    for i in range(l):
        p_prev = contours[(i - range_point) % l]
        p_next = contours[(i + range_point) % l]

        midpoint = (p_prev + p_next) / 2.0
        score = np.linalg.norm(contours[i] - midpoint)
        scores[i] = score

    
    scores = cv2.normalize(
        scores,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX
    ).astype(np.uint8)

    heat_map = np.zeros(image.shape).astype(np.uint8)
    for i in range(len(contours)):
        heat_map[contours[i][1]][contours[i][0]] = scores[i][0]
    cv2.imshow("heatmap",heat_map)

    tresh = np.mean(scores)*1.9   # prblem de type c d int8 jcroi 
    indices = np.where(scores > tresh)[0]
    scores_tresh = scores[indices]
    contours_tresh = contours[indices]


    print(tresh)
    vec_3s = np.hstack((contours_tresh, scores_tresh.astype(np.uint32)))  # cjroi que sa regle
    superpxl = np.array([
        [0,0,255],
        [image.shape[1],0,255],
        [0,image.shape[0],255],
        [image.shape[1],image.shape[0],255],
    ],dtype=np.float32)

    def aprox(sPxs,pxs):
        sPxs = np.array(sPxs)
        pxs = np.array(pxs)
        tree = cKDTree(sPxs)
        _, indices = tree.query(pxs)
        res = []
        for i in range(len(sPxs)):
            assigned = pxs[indices == i]
            if len(assigned) > 0:
                res.append(assigned.mean(axis=0))
            else:
                res.append(sPxs[i])
        return np.array(res)
    
    superpxl = aprox(superpxl,vec_3s)



    heat_map_tresh = np.zeros(image.shape).astype(np.uint8)
    coords = np.clip(superpxl[:, :2].astype(int), 0, [image.shape[1]-1, image.shape[0]-1])
    heat_map_tresh[coords[:,1], coords[:,0]] = 255
    cv2.imshow("heatmap_tresh", heat_map_tresh)
    
    return superpxl[:, :2]

def transform(img,init):
    if len(init) != 4:
        print("pas bon nbr de coin")
        return img
    shape = img.shape
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
    T = cv2.getPerspectiveTransform(a ,b)
    img_trans = cv2.warpPerspective(img,T,(shape[1],shape[0]))
    return img_trans



img_treshold,contours,contours_plaque = image_treatment(image)
corner = detection_corner2(contours_plaque,12)
img_transformed = transform(image,corner)

cv2.imshow("image",image)
cv2.imshow("image contour",cv2.polylines(np.zeros(image.shape),contours_plaque,1,(255,255,255),1 ))
img_corner = np.zeros(image.shape).astype(np.uint8)
index = 50
for (x, y) in corner:
    cv2.circle(
        img_corner,
        (int(x), int(y)),
        radius=1,
        color=index, 
        thickness=1
    )
    index+=50
cv2.imshow("image corner size : "+str(len(corner)),img_corner)
cv2.imshow("image transformed",img_transformed)
#cv2.imwrite("test.png",black_img)
cv2.waitKey(0)
cv2.destroyAllWindows()