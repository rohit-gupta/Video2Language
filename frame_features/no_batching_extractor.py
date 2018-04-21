from keras.layers import TimeDistributed
from keras.layers import Input, LSTM, Dense
from keras.models import Model
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.layers.core import Lambda
from keras import backend as K
from __future__ import print_function

import numpy as np
from glob import glob
import pickle


# add a layer that returns the average over timesteps
def timestepaverage(x):
    x = K.mean(x, axis=1)
    x = K.l2_normalize(x, axis=1)
    return x

def timestepaverage_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 3  # (Batch Size x Time Steps x Feature Size)
    return tuple((shape[0],shape[2]))

# TimeStepAverage = Lambda(timestepaverage, output_shape=timestepaverage_output_shape)

def preprocess_image(img_path):
	img = image.load_img(img_path,	 target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	return x

NUM_FRAMES = 16
FRAME_RATE = 2

# Load Video Meta-data
fname = "video_fps.csv"
with open(fname) as f:
    content = f.readlines()

video_fps_list = [(x.strip()).split(";")[:2] for x in content]

video_fps = {}
for x in video_fps_list:
	video_fps[x[0].split(".")[0]] = float(x[1])

video_folder = "YouTube2Text/youtubeclips-dataset/"
frames_folder = video_folder + "frames/vid*"
videos = glob(frames_folder)


video_frames = []
video_names = []

count = 0

for video in sorted(videos):
	if count % 100 == 0:
		print(int(100*round(count/float(len(videos)),2)), "% Data Loaded")
	video_name = (video.split("/")[-1]).split(".")[0] # Extract video name from path
	fps = video_fps[video_name]
	frame_files = sorted(glob(video +"/*.jpg"))
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
		if NUM_FRAMES/len(frame_data) > 1: # Video is < 1/2 of 8 Seconds
			num_repeats = NUM_FRAMES/len(frame_data) - 1
			for _ in range(num_repeats):
				for itr in range(len(frame_data[:actual_frame_length])):
					frame_data.append(frame_data[:actual_frame_length][itr])
		dup_frame_length = len(frame_data)
		if NUM_FRAMES/len(frame_data) == 1 and NUM_FRAMES > len(frame_data): # Video is slightly smaller than 8 Seconds
			for itr in range(0, NUM_FRAMES -len(frame_data)):
				frame_data.append(frame_data[itr])
	if len(frame_data) != NUM_FRAMES:
		print(og_frame_length, num_repeats, dup_frame_length, len(frame_data))
		raise Exception('Incorrect number of frames sampled')
	frame_data = np.array(frame_data)
	video_frames.append(frame_data)
	video_names.append(video_name)
	count += 1

video_frames = np.array(video_frames)

# Workaround
K.set_learning_phase(1)

video_input = Input(shape=(NUM_FRAMES, 224, 224, 3))
convnet_model = ResNet50(weights='imagenet', include_top=False, pooling='max')
encoded_frame_sequence = TimeDistributed(convnet_model)(video_input)
# encoded_video = TimeStepAverage()(encoded_frame_sequence)
encoded_video = Lambda(timestepaverage, output_shape=timestepaverage_output_shape)(encoded_frame_sequence)

feature_extract_model = Model(inputs=video_input, outputs=encoded_video)
feature_extract_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
feature_extract_model.summary()

predicted_features = feature_extract_model.predict(video_frames, batch_size=16)

video_feature_vectors = {}

for idx,video in enumerate(video_names):
	video_feature_vectors[video] = predicted_features[idx]

pickle.dump(video_feature_vectors, open("average_frame_features.pickle", "wb"))
