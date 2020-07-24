import pandas as pd
import numpy as np
import cv2
from faces_clustering import FeatureExtractor
from sklearn.preprocessing import MinMaxScaler


class FaceSearcher:

    def __init__(self, face_embs, classes_col, distance_col=None, backbone='senet50'):
        self.face_embs = face_embs.copy()
        self.classes_col = classes_col
        self.distance_col = distance_col

        self.extractor = FeatureExtractor(backbone=backbone)
        self.scaler = MinMaxScaler()

        self.cluster_embeddings = pd.DataFrame(
            self.scaler.fit_transform(self.face_embs['embeddings'].values.tolist()),
            index=self.face_embs.index)
        self.cluster_embeddings[classes_col] = self.face_embs[classes_col]
        self.centroids = self.cluster_embeddings.groupby([classes_col]).mean()
        self.classes = self.face_embs[classes_col].unique()

        #if distance_col is None:
        #    self.calculate_distance()

    def calculate_distance(self):
        diffs = []
        for i in self.classes:
            diff = self.cluster_embeddings.loc[self.cluster_embeddings[self.classes_col] == i].iloc[:, :-1].sub(
                self.centroids.iloc[i, :])
            diffs.append(diff.pow(2).mean(axis=1).pow(1 / 2))
        self.distance_col = 'd_'+self.classes_col
        self.face_embs[self.distance_col] = pd.concat(diffs)

    def closest_centroids(self, filename = None, embs_query = None):
        faces_query = None
        if embs_query is None:
            embs_query, faces_query = self.extractor.get_embeddings(filename)
        assert str(embs_query) != '-', 'no face detected'
        embs_query_scaled = self.scaler.transform(embs_query)

        ans = []
        for query in embs_query_scaled:
            all_distance = self.centroids.sub(query).pow(2).mean(axis=1).pow(1 / 2)
            ans.append(all_distance)

        return faces_query, ans

    def closest_faces(self, filename):
        embs_query, faces_query = self.extractor.get_embeddings(filename)
        assert str(embs_query) != '-', 'no face detected'
        embs_query_scaled = self.scaler.transform(embs_query)

        ans = []
        for query in embs_query_scaled:
            all_distance = self.cluster_embeddings.iloc[:, :-1].sub(
                query).pow(2).mean(axis=1).pow(1 / 2)
            ans.append(all_distance)

        return faces_query, ans
