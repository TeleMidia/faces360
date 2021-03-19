import cv2
import os
import pandas as pd
import numpy as np
from faces_clustering import get_files_folder, FeatureExtractor, is_image
from tqdm.notebook import tqdm
from faces_clustering import silhouette, generate_colors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import shutil

class VideoClustering:

	def __init__(self, backbone, detector, clustering_alg, verbose = 0):
		self.verbose = verbose
		self.colors = None
		self.dir_path = None
		self.backbone = backbone
		self.clustering_alg = clustering_alg
		self.cluster_column = f'cluster_{self.clustering_alg}'
		self.extractor = FeatureExtractor(backbone, detector)

	def cluster(self, video_path, fps=None, dir_path = None):
		assert os.path.isfile(video_path), 'video not found'

		if self.verbose > 0:
			print(f'Processing {video_path}')
		#extracting the frames of the video
		if self.verbose > 1:
			print('extracting frames')
		dir_path = self.extract_frames(video_path, fps, dir_path)
		
		#extracting the faces, bounding boxes and embeddings
		if self.verbose > 1:
			print('extracting embeddings')
		dt_embs = self.extract_face_embeddings(dir_path)
		
		#only valid embeddings
		valid = dt_embs.embeddings.apply(lambda x: str(x) != '-')
		dt_embs = dt_embs.loc[valid]

		embs = [list(emb) for emb in dt_embs.embeddings.values]

		#clustering
		if self.verbose > 1:
			print('clustering')
		label_clusters = silhouette(embs,alg = self.clustering_alg, verbose = self.verbose)
		dt_embs[self.cluster_column] = label_clusters

		self.dt_embs = dt_embs
		self.faces_samples = self.dt_embs[[self.cluster_column,'faces']].sort_values(
			self.cluster_column).groupby(self.cluster_column).head(1).faces.values
		
		self.cluster_by_frames = self.generate_cluster_by_frames()
		self.video_metadata = self.generate_video_metadata()

		return self.dt_embs

	def generate_video_metadata(self):

		cluster_embeddings = pd.DataFrame(self.dt_embs['embeddings'].values.tolist(),index=self.dt_embs.index)
		cluster_embeddings[self.cluster_column] = self.dt_embs[self.cluster_column]
		metadata = pd.DataFrame(cluster_embeddings.groupby(
			[self.cluster_column]).mean().sort_index().agg(list, axis=1))

		metadata.columns = ['embeddings']
		metadata['frames'] = self.dt_embs.groupby(self.cluster_column)['frames'].apply(list)
		metadata['total_frames'] = self.cluster_by_frames.shape[0]
		metadata['faces_samples'] = self.faces_samples
		metadata['video'] = self.video_path

		return metadata

		

	def show_people_video(self, colors = None):
		if colors is not None:
			self.colors = colors
		elif self.colors is None:
			self.colors = generate_colors(len(self.faces_samples))


		fig, axes = plt.subplots(nrows=1, ncols=max(len(self.faces_samples),8), figsize=(12, 2))     
		i = 0
		[axi.set_axis_off() for axi in axes.ravel()]
		for sample in self.faces_samples:
			#axes.figure(figsize=(2,3))
			image = cv2.rectangle(sample.copy(), (0,0), sample.shape[0:2], self.colors[i], int(sample.shape[0]/10))

			axes[i].set_title(f'Person {i}')
			axes[i].imshow(image[:,:,::-1])
			axes[i].axis('off')
			axes[i].set_aspect('equal')
			i = i+1
		return fig

	def draw_clusters(self, frame, clusters, bounds, lines = True):
		if self.colors is None:
			self.colors = generate_colors(len(self.faces_samples))

		thumb = f'{self.dir_path}/frame_{frame}.jpg'
		img = cv2.imread(thumb)
		test = img[:,:,::-1].copy()
		for c,b in zip(clusters, bounds):
		  
			if lines:
				line_thickness = int((img.shape[0]/3)/len(self.faces_samples))
				test = cv2.line(test, (0,test.shape[0]-line_thickness*(c+1)),
					(img.shape[1]-1,test.shape[0]-line_thickness*(c+1)), self.colors[c], thickness=line_thickness)
			else:
				start_point = (b[0],b[2])
				end_point = (b[1],b[3])				
				test = cv2.rectangle(test, start_point, end_point, self.colors[c], int(img.shape[0]/100))

		return test

	def save_tagged_frames(self, colors = None, lines = False):
		tag_path = self.dir_path+'_tagged'

		if os.path.isdir(tag_path):
			shutil.rmtree(tag_path)
		os.mkdir(tag_path)

		self.show_people_video(colors)

		i = 0
		for frame in tqdm(self.cluster_by_frames.index.values):
			thumb = self.draw_clusters(frame, self.cluster_by_frames.loc[frame][0], self.cluster_by_frames.loc[frame][1], lines=lines)          
			cv2.imwrite(f'{tag_path}/frame_{frame}.jpg',thumb[:,:,::-1])

	def display_timeline(self, colors = None, lines = True, limit = None):
		if limit is None:
			cluster_by_frames = self.cluster_by_frames
		else:
			cluster_by_frames = self.cluster_by_frames.head(limit)

		row_number = int(np.ceil(cluster_by_frames.shape[0]/6))

		self.show_people_video(colors)

		plt.figure(figsize = (15,row_number*2))
		gs1 = gridspec.GridSpec(row_number, 6)
		gs1.update(wspace=0, hspace=0) # set the spacing between axes. 
		i = 0
		for frame in tqdm(cluster_by_frames.index.values):
			thumb = self.draw_clusters(frame, cluster_by_frames.loc[frame][0], cluster_by_frames.loc[frame][1], lines=lines)
			ax1 = plt.subplot(gs1[i])  
			plt.subplots_adjust(hspace=0, wspace=0)
			ax1.imshow(thumb)
			ax1.axis('off')
			ax1.set_aspect('equal')

			i = i + 1

	def generate_cluster_by_frames(self):
		self.colors = None
		self.dt_embs['frames'] = self.dt_embs.urls.apply(lambda x: int(x.split('.')[0].split('/')[-1].split('_')[-1]))
		frames = [int(x.split('.')[0].split('/')[-1].split('_')[-1]) for x in self.frames_url]
		clusters_frames = self.dt_embs[[self.cluster_column, 'bounds', 'frames']].copy()

		cluster_by_frames = clusters_frames.groupby('frames')[self.cluster_column].apply(list)
		bounds_by_frames = clusters_frames.groupby('frames')['bounds'].apply(list)

		cluster_by_frames = pd.DataFrame(cluster_by_frames)
		cluster_by_frames['bounds'] = bounds_by_frames
		#adding frames without faces
		for f in frames:
			if f not in cluster_by_frames.index.values:
				cluster_by_frames.loc[f] = [[],[]]

		return cluster_by_frames

	def extract_frames(self, video_path, fps=None, dir_path = None):
		if dir_path is None:
			dir_path = video_path.split('.')[0]

		self.dir_path = dir_path
		self.video_path = video_path
		cap=cv2.VideoCapture(video_path)
		original_fps = int(round(cap.get(cv2.CAP_PROP_FPS)))

		#relative fps
		if fps is None:
			fps = original_fps
		else:
			fps = original_fps/fps
		if self.verbose > 1:
			print(f'Original video fps is {original_fps}. Extracting at each {fps} frames')

		if os.path.isdir(dir_path):
			if self.verbose > 1:
				print('Frames already extracted.')
			 #shutil.rmtree(dir_path)
		else:
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

	def extract_face_embeddings(self, dir_path, save_dataframe = False):
		frames_url = get_files_folder(dir_path, is_image)
		self.frames_url = frames_url
		faces_dict = {}
		if self.verbose > 0:
			for url in tqdm(frames_url):
				faces_dict[url] = self.extractor.get_embeddings(url)
		else:
			for url in frames_url:
				faces_dict[url] = self.extractor.get_embeddings(url)

		all_urls = []
		all_faces = []
		all_embs = []
		all_bounds = []
		for url in frames_url:
			embs, faces, bounds = faces_dict[url]
			for emb, face, bound in zip(embs, faces,bounds):
				all_urls.append(url)
				all_faces.append(face)
				all_embs.append(emb)
				all_bounds.append(bound)

		dt_embs = pd.DataFrame(all_urls, columns=['urls'])
		dt_embs['embeddings'] = all_embs
		dt_embs['faces'] = all_faces
		dt_embs['bounds'] = all_bounds

		if save_dataframe:
			dt_embs.to_pickle(f'{dir_path}.pkl')

		return dt_embs 
