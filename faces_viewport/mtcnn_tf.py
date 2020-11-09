from mtcnn import MTCNN
import numpy as np
import cv2

class MTCNN_tf:

	def __init__(self, threshold = [0.7]):
		self.threshold = threshold
		self.detector = MTCNN()

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
				face = pixels[y1:y2, x1:x2]

				if face.shape[0] > 0 and face.shape[1] > 0:
					faces.append(face)
					bounds.append((x1,x2,y1,y2))
					confidences.append(result['confidence'])
					pixels = cv2.rectangle(pixels, (x1,y1), (x2,y2), (255,0,0), 5)
					
		return pixels, bounds, confidences

