from .mtcnn_tf import MTCNN_tf
import sys
sys.path.append('../')

import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import cv2
import os
from utils import *
from . import Equirec2Perspec as E2P
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
		try:
			M = B[m]
		except:
			print('cofs ',S)
			print('lenB ',len(B))
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

class ViewportsFaceDetector():

	def __init__(self, rows = 3, cols = 6, fovw = 60, fovh = 60, width = 720, verbose = 0, torch = False, nms_th = 0.5):
		self.rows = rows
		self.cols = cols
		self.fovw = fovw
		self.fovh = fovh
		self.width = width
		self.verbose = verbose
		self.nms_th = nms_th

		if not torch:
			self.detector = MTCNN_tf()
		else:
			self.detector = MTCNN_Torch()

	def detect_faces_polys(self, path):

		equ, eq_bounds, all_confs = self.detect_faces_viewports(path)
		eq_img = np.uint16(equ._img)
		if self.verbose > 0:
			img = show_bounds(eq_img, eq_bounds)

		adj_bounds = [adjust_bounds(eq_bound.copy(), equ._img.shape[1]) for eq_bound in eq_bounds]

		polys = [geometry.Polygon(adj_bound).buffer(0) for adj_bound in adj_bounds]

		D = non_maximum_supression(polys, all_confs.copy(), self.nms_th)

		org_bounds = [eq_bounds[d] for d in D]
		adj_bounds = [adj_bounds[d] for d in D]
		nms_polys = [polys[d] for d in D]

		if self.verbose > 0:
			img = show_bounds(eq_img, adj_bounds)

			for poly in nms_polys:
			    plt.plot(*poly.exterior.xy)
			    plt.xlim(0, 1.5*img.shape[1])
			    plt.ylim(0, img.shape[0])
			    plt.gca().invert_yaxis()

			plt.show()
		return org_bounds, adj_bounds, nms_polys #original bounds, adjusted bounds (to construct polys), and polys

	def detect_faces_viewports(self, img_path):
		#all_bounds = []
		equ = E2P.Equirectangular(img_path)

		eq_bounds = []
		all_confs = [] #all confidences from detected faces in all lat long coordinates

		if self.verbose > 0:
			fig, axes = plt.subplots(nrows=self.rows, ncols=self.cols, figsize=(18, 10))
		step_lat = -180/(self.rows)
		step_long = 360/(self.cols)
		for i in range(self.rows):
			for j in range(self.cols):
				lat = 90+i*step_lat
				long = -180+j*step_long
				img, long_map, lat_map = equ.GetPerspective(self.fovw, self.fovh, long, lat, self.width)          
				img, bounds, confidences = self.detector.detect_faces_cv2(img) #x1,x2,y1,y2       
				
				border_view = abs(long)+self.fovw/2>=180#true if viewport starts at one side and end in another	        
			  
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
				all_confs = all_confs+confidences
				#all_bounds = all_bounds+bounds

				if self.verbose > 0:
					axes[i,j].set_title(f'Lat: {lat}° Long {long}°')
					axes[i,j].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
					axes[i,j].axis('off')
		if self.verbose > 0:
			plt.show()
			print('Number of bounds', len(eq_bounds))
			print('Confs: ', all_confs)
		return equ, eq_bounds, all_confs
