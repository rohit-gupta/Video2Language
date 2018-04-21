from __future__ import print_function


from keras.utils import Sequence
import json
import re
import pickle
import argparse
from glob import glob

import numpy as np
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input

def preprocess_image(img_path):
	img = image.load_img(img_path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	return x


NUM_FRAMES = 40
FRAME_RATE = 2
BATCH_SIZE = 16

def load_videos(video_ids, video_folder, video_fps_dict):
	frames = []
	for video_name in video_ids:
		fps = video_fps_dict[video_name]
		frame_files = sorted(glob(video_folder + video_name +"/*.jpg"))
		num_frames = len(frame_files)
		gap = int(round(fps/FRAME_RATE)) # If FPS = 30, FRAME_RATE = 3, Frames at ID 0,10,20,30 ... are sampled
		frame_data = []
		for idx,frame_file in enumerate(frame_files):
			if len(frame_data) >= NUM_FRAMES:
				break
			if idx%gap == 0:
				frame_data.append(preprocess_image(frame_file)[0])
		actual_frame_length = len(frame_data)
		# If Video is shorter than 8 seconds repeat the short video
		if len(frame_data) < NUM_FRAMES:
			if NUM_FRAMES/len(frame_data) > 1: # Video is < 1/2 of 10 Seconds
				num_repeats = NUM_FRAMES/len(frame_data) - 1
				for _ in range(num_repeats):
					for itr in range(len(frame_data[:actual_frame_length])):
						frame_data.append(frame_data[:actual_frame_length][itr])
			dup_frame_length = len(frame_data)
			if NUM_FRAMES/len(frame_data) == 1 and NUM_FRAMES > len(frame_data): # Video is slightly smaller than 10 Seconds
				for itr in range(0, NUM_FRAMES -len(frame_data)):
					frame_data.append(frame_data[itr])
		if len(frame_data) != NUM_FRAMES:
			print(actual_frame_length, num_repeats, dup_frame_length, len(frame_data))
			raise Exception('Incorrect number of frames sampled')
		frame_data = np.array(frame_data)
		frames.append(frame_data)
	return np.array(frames)


# Here, `videos` is list of path to the videos
class MSVDSequence(Sequence):
	def __init__(self, video_names, video_folder,fps_dict, batch_size):
		self.batch_size = batch_size
		self.video_folder = video_folder
		self.fps_dict = fps_dict
		self.videos = video_names

	def __len__(self):
		return len(self.videos) // self.batch_size

	def __getitem__(self,idx):
		batch_videos = self.videos[idx*self.batch_size:(idx+1)*self.batch_size]
		return load_videos(batch_videos,self.video_folder,self.fps_dict)

	def on_epoch_end(self):
		pass



# Load Video Meta-data
fname = "../video_fps.csv"
with open(fname) as f:
    content = f.readlines()

video_fps_list = [(x.strip()).split(";")[:2] for x in content]

video_fps = {}
for x in video_fps_list:
	video_fps[x[0].split(".")[0]] = float(x[1])

videos_folder = "../YouTube2Text/youtubeclips-dataset/frames/"

frames_folder = videos_folder + "vid*"
videos = glob(frames_folder)
videos = sorted(videos)

# Load Captions Data
# annotations_file = "train_val_videodatainfo.json"
# with open(annotations_file) as data_file:
# 			train_data = json.load(data_file)

# Load Captions Data
# test_annotations_file = "test_videodatainfo.json"
# with open(test_annotations_file) as data_file:
# 			test_data = json.load(data_file)

# print(len(train_data['sentences']), len(test_data['sentences']))

# captions = train_data['sentences'] + test_data['sentences']
# sorted_captions = sorted(captions, key=lambda k:  int(re.split('(\d+)',k['video_id'])[-2]))
# video_names = sorted(list(set([x['video_id'] for x in sorted_captions])), key=lambda x: int(re.split('(\d+)',x)[-2]))

video_names = [(video_path.split("/")[-1]) for video_path in videos]

# Create generator
all_videos_generator = MSVDSequence(video_names, video_folder=videos_folder, fps_dict=video_fps, batch_size=BATCH_SIZE)

NUM_STEPS = len(all_videos_generator)

print("Frames will be extracted for", NUM_STEPS * BATCH_SIZE, "Videos")


from keras.applications.resnet50 import ResNet50
from keras.layers import TimeDistributed, Bidirectional
from keras.layers import Input, LSTM, Dense
from keras.models import Model
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from keras.layers.core import Lambda

from keras import backend as K
K.set_learning_phase(1)

# add a layer that returns the average over timesteps
def timestepaverage(x):
    x = K.mean(x, axis=1)
    x = K.l2_normalize(x, axis=1)
    return x

def timestepaverage_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 3  # (Batch Size x Time Steps x Feature Size)
    return tuple((shape[0],shape[2]))


# Define Model
video_input = Input(shape=(NUM_FRAMES, 224, 224, 3))
convnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
for layer in convnet_model.layers:
    layer.trainable = False

encoded_frame_sequence = TimeDistributed(convnet_model)(video_input)
encoded_video = Lambda(timestepaverage, output_shape=timestepaverage_output_shape)(encoded_frame_sequence)

feature_extract_model = Model(inputs=video_input, outputs=encoded_video)
feature_extract_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
feature_extract_model.summary()

predicted_features = feature_extract_model.predict_generator(all_videos_generator, NUM_STEPS, max_queue_size=20, use_multiprocessing=True, verbose=1)

video_feature_vectors = {}

for idx,video in enumerate(video_names):
	if idx < len(predicted_features):
		video_feature_vectors[video] = predicted_features[idx]

pickle.dump(video_feature_vectors, open("average_frame_features.pickle", "wb"))
