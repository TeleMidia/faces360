import os
from matplotlib import pyplot as plt
import cv2
import numpy as np


def is_image(file):
    return file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))

def is_video(file):
    return file.lower().endswith(('.mp4', '.avi'))    

def get_files_folder(folder, criteria=lambda x: True):
    complete_urls = []
    for a, b, c in os.walk(folder):
        complete_urls += [os.path.join(a, x) for x in c if criteria(x)]
    return complete_urls


def display_image(url):
    plt.figure()
    plt.imshow(cv2.imread(url)[:, :, ::-1])

def generate_colors(n):
    color_values = []
    h = 0
    s = 255
    v = 255
    steph = 180 / n
    for _ in range(n):        
        aux = np.uint8(np.array([h,s,v]).reshape((1,1,3)))
        #print(aux)
        color = cv2.cvtColor(aux, cv2.COLOR_HSV2RGB).reshape(3).astype(float)
        color_values.append(color)
        h += steph

    return color_values

def adjust_bounds(vec, size_img = 1280):
    #print(vec)
    right_side = [a for a in vec if a[0]>size_img//2]
    if len(right_side) > 0 and len(right_side) < len(vec):
        mean_right_x = np.mean(right_side, axis=0)[0]
        for i in range(len(vec)):
            if vec[i] not in right_side:
                if abs(vec[i][0]+size_img-mean_right_x)<abs(vec[i][0]-mean_right_x):
                    vec[i] = (vec[i][0]+size_img, vec[i][1])
    return vec