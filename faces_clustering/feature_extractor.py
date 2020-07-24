import os
import pandas as pd
import numpy as np
import cv2
from mtcnn import MTCNN
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace


class FeatureExtractor:
    """docstring for FeatureExtractor"""

    # backbone=['vgg16', 'resnet50', 'senet50']
    def __init__(self, backbone):
        self.backbone = backbone
        self.detector = MTCNN()
        self.model = VGGFace(model=self.backbone, include_top=False, input_shape=(224, 224, 3), pooling='avg')

    def extract(self, urls):
        """
        urls: urls of the images
        returns one face embedding per url
        """
        self.df_imgs = pd.DataFrame(urls, columns=['urls'])
        embeddings = self.df_imgs.urls.apply(lambda x: self.get_embeddings(x)[0][0])
        self.df_imgs['embeddings'] = embeddings

        return self.df_imgs

    def extract_faces(self, filename, required_size=(224, 224), confidence=0.9):
        pixels = cv2.imread(filename)
        if pixels is not None:
            pixels_rgb = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)

            results = self.detector.detect_faces(pixels_rgb)

            faces = []
            bounds = []
            for result in results:
                if result['confidence'] >= confidence:
                    x1, y1, width, height = result['box']
                    x2, y2 = x1 + width, y1 + height
                    x1 = max(x1,0)
                    y1 = max(y1,0)
                    face = pixels_rgb[y1:y2, x1:x2]

                    if face.shape[0] > 0 and face.shape[1] > 0:
                        faces.append(cv2.resize(face, required_size))
                        bounds.append((x1,x2,y1,y2))
            if len(faces) > 0:
                return faces, bounds
        return ('no_face','no_face')

    def get_embeddings(self, filename):
        faces, bounds = self.extract_faces(filename)
        if str(faces) != 'no_face':
            # print('face')
            sample = np.asarray(faces, 'float32')
            # sample = np.expand_dims(sample, axis=0)
            sample = preprocess_input(sample, version=2)
            embedding = self.model.predict(sample)
            return embedding, faces, bounds
        else:
            # print(face)
            return ['-']*3
