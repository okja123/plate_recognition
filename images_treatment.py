import kagglehub
import os
import cv2

def download_plate():
    os.environ['KAGGLEHUB_CACHE'] = "./"
    path = kagglehub.dataset_download("abdelhamidzakaria/european-license-plates-dataset")
    print("Path to dataset files:", path)

PATH_TO_DATASET = "./datasets/abdelhamidzakaria/european-license-plates-dataset/versions/1/dataset_final/test"
img = cv2.imread(PATH_TO_DATASET, -1)

