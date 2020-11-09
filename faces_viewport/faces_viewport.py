from .mtcnn_tf import MTCNN_tf
import sys
sys.path.append('../')

import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import cv2
import os
from faces_clustering import get_files_folder, VideoClustering, is_image
from faces_clustering import Equirec2Perspec as E2P
from mtcnn_torch import MTCNN_Torch
import numpy as np
from shapely import geometry

def extract_frames(video_path):
	dir_path = video_path.split('.')[0]

	cap=cv2.VideoCapture(video_path)
	fps = int(round(cap.get(cv2.CAP_PROP_FPS)))

	if os.path.isdir(dir_path):
		os.rmdir(dir_path)
	os.mkdir(dir_path)

	i=1
	while(cap.isOpened()):
		ret, frame = cap.read()
		if ret == False:
			break
		if i%fps == 0:
			cv2.imwrite(f'{dir_path}/frame_{i}.jpg',frame)
		i+=1

	return dir_path

'''
def detect_faces_tf(detector, pixels):    
	
	results = detector.detect_faces(pixels)
	
	faces = []
	bounds = []
	confidences = []
	for result in results:
		if result['confidence'] >= 0.7:
			x1, y1, width, height = result['box']
			x2, y2 = x1 + width, y1 + height
			x1 = max(x1,0)
			y1 = max(y1,0)
			x2 = min(x2,pixels.shape[1]-1)
			y2 = min(y2,pixels.shape[0]-1)
			face = pixels[y1:y2, x1:x2]

			if face.shape[0] > 0 and face.shape[1] > 0:
				faces.append(face)
				bounds.append((x1,x2,y1,y2))
				confidences.append(result['confidence'])
				pixels = cv2.rectangle(pixels, (x1,y1), (x2,y2), (255,0,0), 5)
				
	return pixels, bounds, confidences

def add_point(points, new_point, border_view, width_eq, fovw):
	
	if border_view:
		thresh = int(width_eq*fovw/360)-2
		if new_point[0] < thresh:
			#print("entrou")
			new_point = (new_point[0]+width_eq, new_point[1])

	if new_point not in points:
		points = points+[new_point]
	return points
'''

def detect_faces_viewports(img_path, rows = 4, cols = 9, fovw = 60, fovh = 60, width = 720, verbose = 0):
	#all_bounds = []
	equ = E2P.Equirectangular(img_path)

	eq_bounds = []
	all_confs = [] #all confidences from detected faces in all lat long coordinates
	detector = MTCNN_tf()
	#detector = MTCNN_Torch()
	if verbose > 0:
		fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(18, 10))


	for i in range(rows):
		for j in range(cols):
			lat = i*(-60)+90
			long = -180+45*j
			img, long_map, lat_map = equ.GetPerspective(fovw, fovh, long, lat, width)    
			#img, bounds, confidences = detect_faces_tf(detector, np.uint16(img)) #x1,x2,y1,y2       
			img, bounds, confidences = detector.detect_faces_cv2(img) #x1,x2,y1,y2       
			
			border_view = abs(long)+fovw/2>=180#true if viewport starts at one side and end in another	        
		  
			for bound in bounds:
				x1, x2, y1, y2 = bound
				x1, x2 = min([x1,x2]),max([x1,x2])
				y1, y2 = min([y1,y2]),max([y1,y2])
				points = []
				'''
				for x in range(x1,x2+1):
					new_point = (int(long_map[y1, x]), int(lat_map[y1, x]))                
					points = add_point(points, new_point, border_view, equ._img.shape[1], fovw)
				for y in range(y1,y2+1):
					new_point = (int(long_map[y, x2]), int(lat_map[y, x2]))
					points = add_point(points, new_point, border_view, equ._img.shape[1], fovw)
				for x in range(x2,x1-1,-1):
					new_point = (int(long_map[y2, x]), int(lat_map[y2, x]))
					points = add_point(points, new_point, border_view, equ._img.shape[1], fovw)
				for y in range(y2,y1-1,-1):
					new_point = (int(long_map[y, x1]), int(lat_map[y, x1]))
					points = add_point(points, new_point, border_view, equ._img.shape[1], fovw)
				'''
				points.append((int(long_map[y1, x1]), int(lat_map[y1, x1])))
				points.append((int(long_map[y1, (x1+x2)//2]), int(lat_map[y1, (x1+x2)//2])))
				points.append((int(long_map[y1, x2]), int(lat_map[y1, x2])))

				points.append((int(long_map[(y1+y2)//2, x2]), int(lat_map[(y1+y2)//2, x2])))

				points.append((int(long_map[y2, x2]), int(lat_map[y2, x2])))
				points.append((int(long_map[y2, (x1+x2)//2]), int(lat_map[y2, (x1+x2)//2])))
				points.append((int(long_map[y2, x1]), int(lat_map[y2, x1])))

				points.append((int(long_map[(y1+y2)//2, x1]), int(lat_map[(y1+y2)//2, x1])))

				#points = adjust_bounds(points, equ._img.shape[1])

				eq_bounds = eq_bounds+[points]
				all_confs = all_confs+[confidences]
			#all_bounds = all_bounds+bounds

			if verbose > 0:
				axes[i,j].set_title(f'Lat: {lat}° Long {long}°')
				axes[i,j].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
				axes[i,j].axis('off')
	if verbose > 0:
		plt.show()

	return equ, eq_bounds, all_confs

def show_bounds(eq_img, bounds):	
	img = eq_img.copy()
	for eq_bound in bounds:	    
		for point in eq_bound:
			adj_point = (point[0]%eq_img.shape[1], point[1]%eq_img.shape[0])
			cv2.circle(img, adj_point, 4, (255, 0, 0), -1)
			
	plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
	plt.show()	

	return img

def non_maximum_supression(B, S, nt):
	D = []
	B_back = B
	B = B.copy()
	while len(B) > 0:
		m = np.argmax(S)
		M = B[m]
		D.append(B_back.index(B[m]))
		B.pop(m)
		S.pop(m)
		
		to_remove = []
		for i in range(len(B)):
			iou = M.intersection(B[i]).area/M.union(B[i]).area
			if iou >= nt:
				to_remove.append(i)
				
		for i in sorted(to_remove, reverse=True):
			B.pop(i)
			S.pop(i)
	return D