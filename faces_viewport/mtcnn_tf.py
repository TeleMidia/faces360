from mtcnn import MTCNN
import numpy as np
import cv2
from utils import *
from shapely import geometry
import matplotlib.pyplot as plt

class MTCNN_tf:

	def __init__(self, threshold = [0.7], verbose = 0):
		self.threshold = threshold
		self.detector = MTCNN()
		self.verbose = verbose

	def detect_faces_cv2(self, img):
		pixels = np.uint16(img)

		results = self.detector.detect_faces(pixels)
	
		faces = []
		bounds = []
		confidences = []
		for result in results:
			if result['confidence'] >= self.threshold[-1]:
				x1, y1, width, height = result['box']
				x2, y2 = x1 + width, y1 + height
				x1 = max(x1,0)
				y1 = max(y1,0)
				x2 = min(x2,pixels.shape[1]-1)
				y2 = min(y2,pixels.shape[0]-1)
				face = pixels[y1:y2, x1:x2].copy()

				if face.shape[0] > 0 and face.shape[1] > 0:
					faces.append(face)
					bounds.append((x1,x2,y1,y2))
					confidences.append(result['confidence'])
					pixels = cv2.rectangle(pixels, (x1,y1), (x2,y2), (255,0,0), 5)
					
		return pixels, bounds, confidences, faces

	def detect_faces_polys(self, path):
		img = cv2.imread(path)
		cv2_image, bounds, confidences, faces = self.detect_faces_cv2(img)

		if self.verbose>0:
			plt.imshow(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
			plt.show()

		eq_bounds = []
		for bound in bounds:
			x1, x2, y1, y2 = bound
			x1, x2 = min([x1,x2]),max([x1,x2])
			y1, y2 = min([y1,y2]),max([y1,y2])
			points = []
			
			points.append((int(x1), int(y1)))
			points.append((int(x2), int(y1)))
			points.append((int(x2), int(y2)))
			points.append((int(x1), int(y2)))

			#points = adjust_bounds(points, equ._img.shape[1])

			eq_bounds = eq_bounds+[points]

		adj_bounds = [adjust_bounds(eq_bound.copy(), img.shape[1]) for eq_bound in eq_bounds]
		polys = [geometry.Polygon(adj_bound).buffer(0) for adj_bound in adj_bounds]

		return eq_bounds, adj_bounds, polys, confidences, faces
