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