import re
import cv2
import numpy as np

float_str2int = lambda x: int(round(float(x)))

def face_eliptical2rectangle(face):
    theta = np.radians(face['angle'])
    a = face['axes_length'][0]
    b = face['axes_length'][1]

    x = np.sqrt(a**2*np.cos(theta)**2+b**2*np.sin(theta)**2)
    y = np.sqrt(a**2*np.sin(theta)**2+b**2*np.cos(theta)**2)

    x_min = float_str2int(face['center'][0] - x)
    x_max = float_str2int(face['center'][0] + x)

    y_min = float_str2int(face['center'][1] - y)
    y_max = float_str2int(face['center'][1] + y)
    
    return (x_min, y_min), (x_max, y_max)

def read_fold(file_path):

	file = open(file_path, 'r')
	line = file.readline()

	images = []

	while line!='':
	    im = {}
	    line = line.rstrip('\n')
	    
	    im['path'] = line
	    faces_n = int(file.readline().rstrip('\n'))
	    
	    faces = []
	    for i in range(faces_n):
	        line = file.readline().rstrip('\n')
	        major, minor, angle, center_x, center_y, _ = re.split(' +', line)
	        
	        face = {
	            'axes_length': (float_str2int(major), float_str2int(minor)),
	            'angle': np.degrees(float(angle)),
	            'center': (float_str2int(center_x),float_str2int(center_y))
	        }
	        face['bounding_box'] = face_eliptical2rectangle(face)
	        faces.append(face)
	        
	    im['faces'] = faces
	    line = file.readline()    
	    images.append(im)    
	file.close()

	return images

def draw_image(image, draw_faces = True):

	mat = cv2.imread('../data/originalPics/'+image['path']+'.jpg', cv2.IMREAD_COLOR)

	if draw_faces:
		faces = image['faces']

		for face in faces:
		    start_point, end_point = face['bounding_box']
		    
		    mat = cv2.ellipse(mat, face['center'], face['axes_length'], face['angle'], 0, 360,(0,0,255), 2)
		    
		    mat = cv2.rectangle(mat, start_point, end_point, (0,255,0), 2)

	return mat
	