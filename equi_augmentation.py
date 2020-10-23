import cv2
import numpy as np

def polar_to_3d(dphi, dtheta): #in radians

    p_x = np.cos(dphi) * np.cos(dtheta);
    p_y = np.cos(dphi) * np.sin(dtheta);
    p_z = np.sin(dphi);

    p1 = np.array([p_x,p_y,p_z])    
        
    return p1

def get_rotation_matrices(phi, theta): #in radians
    
    ## matrix to rotate. First y, then z
    y_axis = np.array([0.0, 1.0, 0.0], np.float32)*phi
    z_axis = np.array([0.0, 0.0, 1.0], np.float32)*theta

    M1 = cv2.Rodrigues(y_axis)[0]
    M2 = cv2.Rodrigues(z_axis)[0]
    M = np.dot(M1,M2)
    
    # matrix to inverse rotation
    y_axis = np.array([0.0, 1.0, 0.0], np.float32)*(-phi)
    z_axis = np.array([0.0, 0.0, 1.0], np.float32)*(-theta)

    M1_inv = cv2.Rodrigues(z_axis)[0]
    M2_inv = cv2.Rodrigues(y_axis)[0]
    M_inv = np.dot(M1_inv,M2_inv)
    
    return M, M_inv

def get8boundingpointsy_x(bounding_box):#bounding_box = (minx, miny),(maxx,maxy)
    
    (minx, miny),(maxx,maxy) = bounding_box
    middle_x = (minx+maxx)//2
    middle_y = (miny+maxy)//2
    
    t_left   = (miny,minx) #top #left
    t_center = (miny, middle_x) #top center
    t_right  = (miny, maxx) #top #right
    c_left   = (middle_y, minx) #left center
    c_right  = (middle_y, maxx) #right center
    b_left   = (maxy, minx) #bottom #left
    b_center = (maxy, middle_x) #bottom center
    b_right  = (maxy, maxx) #bottom #right
    
    return [t_left, t_center, t_right, c_left, c_right, b_left, b_center, b_right]

def project_points(points, delta, eq_cx, eq_cy, r_h, r_w, M, image):
	#equi_points = equirectangular_image.copy()  

	eq_bound = []

	for p in points:
	    py, px = p
	    py = (py/image.shape[0])*r_h
	    px = (px/image.shape[1])*r_w
	    
	    py = py - r_h/2
	    px = px - r_w/2
	    
	    ptheta = np.arctan(px)
	    pphi = np.arctan(py)
	    
	    p1 = polar_to_3d(pphi, ptheta)
	    
	    p1 = np.dot(p1, M)
	    
	    ntheta = np.degrees(np.arctan2(p1[1], p1[0]))
	    nphi = np.degrees(np.arcsin(p1[2]))
	    
	    x = (ntheta/delta)+eq_cx
	    y = (nphi/delta)+eq_cy
	    x = int(np.round(x))
	    y = int(np.round(y))
	    
	    #equi_points = cv2.circle(equi_points, (x,y), 8, (0,0,255), 5)
	    
	    eq_bound.append((y,x))
	    
	return eq_bound

'''
alpha = tolerance of difference of top and bottom middle points in x
'''
def get_search_ranges(eq_bound, eq_w, eq_h, phi, theta, alpha = 5):
	top = min([y[0] for y in eq_bound[:3]])
	bottom = max([y[0] for y in eq_bound[-3:]])
	left = min([eq_bound[0][1],eq_bound[3][1], eq_bound[5][1]]) #1,4,6
	right = max([eq_bound[2][1],eq_bound[4][1], eq_bound[7][1]])#3,5,8

	top_center = eq_bound[1][1]%(eq_w-1)
	bottom_center = eq_bound[6][1]%(eq_w-1)

	if abs(top_center - bottom_center) < alpha: #they are aligned
	    
	    if right < left:
	        part1 = np.arange(left, eq_w)
	        part2 = np.arange(0, right+1)
	        x_range = np.concatenate((part1,part2))    
	    else:
	        x_range = np.arange(left, right+1)
	        
	    y_range = np.arange(top, bottom+1)

	else: #they are not aligned, close to the poles
	    x_range = np.arange(0, eq_w)
	    
	    if phi > 0: #down
	        y_range = np.arange(top, eq_h)
	    else: #up
	        y_range = np.arange(0, bottom+1)

	return x_range, y_range

def draw_points(image, points, color = (0,0,255), radius = 8, thickness = -1):
	image_points = image.copy()
	for i, p in enumerate(points):	    
	    image_points = cv2.circle(image_points, (p[1],p[0]), radius, color, thickness)
	return image_points

'''
r_h:  height of projected image, assuming that the radius
of the sphere is 1.
phi (radians): vertical angle, down is positive and up is negative
theta (radians): horizontal angle
'''
def image_projection_to_equi(equirectangular_image, image, phi = 0, theta = 0, r_h = 1, draw_intermediate = False):

	eq_h, eq_w, _ =  equirectangular_image.shape

	eq_cx = eq_w // 2.0
	eq_cy = eq_h // 2.0

	delta = 180/eq_h # number of degrees per pixel

	h, w, _ = image.shape

	r_w = (image.shape[1]/image.shape[0])*r_h

	points = get8boundingpointsy_x(((0,0),(w-1,h-1)))#bounding_box = (minx, miny),(maxx,maxy) 

	M, M_inv = get_rotation_matrices(phi, theta)

	points_projector = lambda ps: project_points(ps, delta, eq_cx, eq_cy, r_h, r_w, M, image)

	eq_bound = points_projector(points)

	x_range, y_range = get_search_ranges(eq_bound, eq_w, eq_h, phi, theta)

	if draw_intermediate:
		image_points = draw_points(image, points, radius = 4)
		eq_points = draw_points(equirectangular_image, eq_bound, radius = 4)
		eq_area = equirectangular_image.copy()

	for i in y_range:
	#for i in range(eq_h):
	    for j in x_range:
	    #for j in range(eq_w):
	        
	        dtheta = np.radians((j - eq_cx)*delta)
	        dphi = np.radians((i - eq_cy)*delta)
	        
	        p1 = np.dot(polar_to_3d(dphi, dtheta), M_inv)
	        
	        dtheta = np.arctan2(p1[1], p1[0]);
	        dphi = np.arcsin(p1[2])
	        
	        tanx = np.tan(dtheta) + r_w/2
	        tany = np.tan(dphi) + r_h/2

	        if tanx >= 0 and tanx <= r_w and tany>=0 and tany<=r_h:

	            posx = int((tanx/r_w) * image.shape[1])
	            posy = int((tany/r_h) * image.shape[0])               
	            pixel = image[posy, posx]

	            equirectangular_image[i,j] = pixel	            
	            if draw_intermediate:
	            	 eq_area[i,j] = pixel

	        elif draw_intermediate:
	            eq_area[i,j] = [0, 0, 255]
	
	if draw_intermediate:
		return equirectangular_image, [image_points, eq_points, eq_area], points_projector
	return equirectangular_image, None, points_projector