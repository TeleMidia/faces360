import numpy as np
import torch
from torch.autograd import Variable
from PIL import Image
from .get_nets import PNet, RNet, ONet
from .box_utils import nms, calibrate_box, get_image_boxes, convert_to_square
from .first_stage import run_first_stage
import cv2
from utils import *
from shapely import geometry
import matplotlib.pyplot as plt

class MTCNN_Torch:
    def __init__(self, min_face_size=20.0, thresholds=[0.6, 0.7, 0.8], nms_thresholds=[0.7, 0.7, 0.7], verbose = 0):
        """
        Arguments:
        min_face_size: a float number.
        thresholds: a list of length 3.
        nms_thresholds: a list of length 3.
        """

        self.min_face_size = min_face_size
        self.thresholds = thresholds
        self.nms_thresholds = nms_thresholds
 

        self.pnet = PNet()
        self.rnet = RNet()
        self.onet = ONet()
        self.verbose = verbose
        self.onet.eval()

    def detect_faces(self, image):
        """
        Arguments:
            image: an instance of PIL.Image.
            

        Returns:
            two float numpy arrays of shapes [n_boxes, 4] and [n_boxes, 10],
            bounding boxes and facial landmarks.
        """

        # LOAD MODELS
        

        # BUILD AN IMAGE PYRAMID
        width, height = image.size
        min_length = min(height, width)

        min_detection_size = 12
        factor = 0.707  # sqrt(0.5)

        # scales for scaling the image
        scales = []

        # scales the image so that
        # minimum size that we can detect equals to
        # minimum face size that we want to detect
        m = min_detection_size/self.min_face_size
        min_length *= m

        factor_count = 0
        while min_length > min_detection_size:
            scales.append(m*factor**factor_count)
            min_length *= factor
            factor_count += 1

        # STAGE 1

        # it will be returned
        bounding_boxes = []

        # run P-Net on different scales
        for s in scales:
            boxes = run_first_stage(image, self.pnet, scale=s, threshold=self.thresholds[0])
            bounding_boxes.append(boxes)

        # collect boxes (and offsets, and scores) from different scales
        bounding_boxes = [i for i in bounding_boxes if i is not None]
        if len(bounding_boxes) == 0:
            return bounding_boxes, []
        bounding_boxes = np.vstack(bounding_boxes)

        keep = nms(bounding_boxes[:, 0:5], self.nms_thresholds[0])
        bounding_boxes = bounding_boxes[keep]

        # use offsets predicted by pnet to transform bounding boxes
        bounding_boxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
        # shape [n_boxes, 5]

        bounding_boxes = convert_to_square(bounding_boxes)
        bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

        # STAGE 2

        img_boxes = get_image_boxes(bounding_boxes, image, size=24)
        with torch.no_grad():
            img_boxes = Variable(torch.FloatTensor(img_boxes))
        output = self.rnet(img_boxes)
        offsets = output[0].data.numpy()  # shape [n_boxes, 4]
        probs = output[1].data.numpy()  # shape [n_boxes, 2]

        keep = np.where(probs[:, 1] > self.thresholds[1])[0]
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
        offsets = offsets[keep]

        keep = nms(bounding_boxes, self.nms_thresholds[1])
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes = calibrate_box(bounding_boxes, offsets[keep])
        bounding_boxes = convert_to_square(bounding_boxes)
        bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

        # STAGE 3

        img_boxes = get_image_boxes(bounding_boxes, image, size=48)
        if len(img_boxes) == 0: 
            return [], []
        with torch.no_grad():
            img_boxes = Variable(torch.FloatTensor(img_boxes))
        output = self.onet(img_boxes)
        landmarks = output[0].data.numpy()  # shape [n_boxes, 10]
        offsets = output[1].data.numpy()  # shape [n_boxes, 4]
        probs = output[2].data.numpy()  # shape [n_boxes, 2]

        keep = np.where(probs[:, 1] > self.thresholds[2])[0]
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
        offsets = offsets[keep]
        landmarks = landmarks[keep]

        # compute landmark points
        width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
        height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
        xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
        landmarks[:, 0:5] = np.expand_dims(xmin, 1) + np.expand_dims(width, 1)*landmarks[:, 0:5]
        landmarks[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(height, 1)*landmarks[:, 5:10]

        bounding_boxes = calibrate_box(bounding_boxes, offsets)
        keep = nms(bounding_boxes, self.nms_thresholds[2], mode='min')
        bounding_boxes = bounding_boxes[keep]
        landmarks = landmarks[keep]

        return bounding_boxes, landmarks

    def detect_faces_cv2(self, cv2_image):
        img = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        #img = cv2_image
        im_pil = Image.fromarray(img)

        bounding_boxes, landmarks = self.detect_faces(im_pil)

        faces = []
        bounds = []
        confidences = []

        for bounding in bounding_boxes:
            x1, y1, x2, y2, c =  bounding
            x1, y1, x2, y2    = int(x1), int(y1), int(x2), int(y2)

            x1 = max(x1,0)
            y1 = max(y1,0)
            x2 = min(x2,cv2_image.shape[1]-1)
            y2 = min(y2,cv2_image.shape[0]-1)
            face = cv2_image[y1:y2, x1:x2].copy()

            if face.shape[0] > 0 and face.shape[1] > 0:
                faces.append(face)
                bounds.append((x1,x2,y1,y2))
                confidences.append(c)
                cv2_image = cv2.rectangle(cv2_image, (x1,y1), (x2,y2), (255,0,0), 5)

        return cv2_image, bounds, confidences, faces

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
