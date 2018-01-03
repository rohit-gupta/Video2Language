from keras.applications.vgg16 import VGG16
from keras.layers import TimeDistributed
from keras.layers import Input, LSTM, Dense
from keras.models import Model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.callbacks import CSVLogger, ModelCheckpoint
import numpy as np
from glob import glob
import h5py
import pickle 

import argparse

NUM_FRAMES = 16
FRAME_RATE = 2
def preprocess_image(img_path):
	img = image.load_img(img_path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	return x


video_folder = "../Youtube2Text/youtubeclips-dataset/"

# Load Train/Test split
fname = video_folder + "train.txt"
with open(fname) as f:
    content = f.readlines()

train = [x.strip() for x in content]

fname = video_folder + "test.txt"
with open(fname) as f:
    content = f.readlines()

test = [x.strip() for x in content]

# Load Video Meta-data
fname = "video_fps.csv"
with open(fname) as f:
    content = f.readlines()

video_fps_list = [(x.strip()).split(";")[:2] for x in content]

video_fps = {}
for x in video_fps_list:
	video_fps[x[0].split(".")[0]] = float(x[1])

# Load and Pre-Process Videos
frames_folder = video_folder + "frames/vid*"
videos = glob(frames_folder)

test_frames = []
# train_frames = []

test_list = []
# train_list = []

count = 0

for video in sorted(videos):
	if count % 100 == 0:
		print int(100*round(count/float(len(videos)),2)), "% Data Loaded"
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
		print og_frame_length, num_repeats, dup_frame_length, len(frame_data)
		raise Exception, 'Incorrect number of frames sampled'
	frame_data = np.array(frame_data)
	if video_name in test:
		test_frames.append(frame_data)
		test_list.append(video_name)
#	if video_name in train:
#		train_frames.append(frame_data)
#		train_list.append(video_name)
	count += 1

#train_frames = np.array(train_frames)
test_frames = np.array(test_frames)

action_vectors = pickle.load(open("../action_classifier/action_vectors_long.pickle", "rb"))
# entity_vectors = pickle.load(open("../entity_classifier/entity_vectors_long.pickle", "rb"))
# attribute_vectors = pickle.load(open("../attribute_classifier/attribute_vectors_long.pickle", "rb"))

NUM_ACTIONS = action_vectors[action_vectors.keys()[0]].shape[0]

#train_tags = np.array([action_vectors[video] for video in train_list])
test_tags  = np.array([action_vectors[video] for video in test_list])


# Save to disk
# f_hdf5_train = h5py.File(video_folder + 'train_frames.hdf5', 'w')
# f_hdf5_train.create_dataset('train_frames', data = train_frames, compression = 'gzip', compression_opts = 9, chunks=True)
# f_hdf5_train.close()

# f_hdf5_test = h5py.File(video_folder + 'test_frames.hdf5', 'w')
# f_hdf5_test.create_dataset('test_frames', data = test_frames, compression = 'gzip', compression_opts = 9, chunks=True)
# f_hdf5_test.close()

# Define Model
video_input = Input(shape=(NUM_FRAMES, 3, 224, 224))
convnet_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
for layer in convnet_model.layers:
    layer.trainable = False

encoded_frame_sequence = TimeDistributed(convnet_model)(video_input)
encoded_video = LSTM(256)(encoded_frame_sequence) 
output = Dense(NUM_ACTIONS, activation='sigmoid')(encoded_video)

action_model = Model(inputs=video_input, outputs=output)
action_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
action_model.summary()

# Train Model

# csv_logger = CSVLogger('../logs/lstm_vgg_tag_model.log')
# checkpointer = ModelCheckpoint(filepath='../models/lstm_vgg_tag_model.h5', verbose=1, save_best_only=True)
# action_model.fit(train_frames, train_tags, epochs=20, batch_size=16, validation_split=0.2, callbacks=[csv_logger, checkpointer], implementation=1)



# Measure Model Performance on test set: Hamming Loss, Recall, Precision
action_model.load_weights('../models/lstm_vgg_tag_model.h5')
preds = action_model.predict(test_frames,batch_size=16)


# from sklearn.metrics import hamming_loss

# preds_binarized = preds 
# preds_binarized[preds>=0.5] = 1
# preds_binarized[preds<0.5] = 0

# print "Hamming Loss: ", hamming_loss(test_tags, preds_binarized)


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

# For each class
precision = dict()
recall = dict()
average_precision = dict()
# for i in range(NUM_ACTIONS):
#     precision[i], recall[i], _ = precision_recall_curve(test_tags[:, i], preds[:, i])
#    average_precision[i] = average_precision_score(test_tags[:, i], preds[:, i])

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], thresholds = precision_recall_curve(test_tags.ravel(), preds.ravel())
average_precision["micro"] = average_precision_score(test_tags, preds, average="micro")
print 'Average precision score, micro-averaged over all classes:', average_precision["micro"]

# Plot uAP v Recall curve
import matplotlib.pyplot as plt

plt.figure()
plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2, where='post')
plt.fill_between(recall["micro"], precision["micro"], step='post', alpha=0.2, color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Average precision score, micro-averaged over all classes: AUC={0:0.2f}'.format(average_precision["micro"]))
plt.savefig('PR_Curve.png')

print thresholds
