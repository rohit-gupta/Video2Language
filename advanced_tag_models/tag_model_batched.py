from keras.applications.resnet50 import ResNet50
from keras.layers import TimeDistributed, Bidirectional
from keras.layers import Input, LSTM, Dense
from keras.models import Model
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from keras import regularizers
from __future__ import print_function


import numpy as np
from glob import glob
import h5py
import pickle
import argparse
import tensorflow as tf
import copy

parser = argparse.ArgumentParser()
parser.add_argument('-t', action='store', dest='tag_type', help='(action/entity/attribute) Type of Tags to train model on')
parser.add_argument('-l', action='store', dest='lstm_size', type=int, help='LSTM Hidden State Size')
# parser.add_argument('-d', action='store', dest='gpu', help='GPU to use')
#parser.add_argument('-s', action='store_true', default=False, dest='save_predictions', help='Save predicted tags')

results = parser.parse_args()

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
fname = "../video_fps.csv"
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
train_frames = []

test_list = []
train_list = []

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
		raise Exception, 'Incorrect number of frames sampled'
	frame_data = np.array(frame_data)
	if video_name in test:
		test_frames.append(frame_data)
		test_list.append(video_name)
	if video_name in train:
		train_frames.append(frame_data)
		train_list.append(video_name)
	count += 1

train_frames = np.array(train_frames)
test_frames = np.array(test_frames)


tag_vectors = pickle.load(open("../"+results.tag_type+"_classifier/" + results.tag_type + "_vectors_long.pickle", "rb"))
# entity_vectors = pickle.load(open("../entity_classifier/entity_vectors_long.pickle", "rb"))
# attribute_vectors = pickle.load(open("../attribute_classifier/attribute_vectors_long.pickle", "rb"))

NUM_TAGS = tag_vectors[tag_vectors.keys()[0]].shape[0]

train_tags = np.array([tag_vectors.get(video,np.zeros(NUM_TAGS)) for video in train_list])
test_tags  = np.array([tag_vectors.get(video,np.zeros(NUM_TAGS)) for video in test_list])

# Data Augmentation
flipped_train_frames = train_frames[:,:,::-1,:,:]
augmented_train_frames = np.concatenate((train_frames,flipped_train_frames))
augmented_train_tags = np.concatenate((train_tags,train_tags))

flipped_test_frames = test_frames[:,:,::-1,:,:]

# Save to disk
# f_hdf5_train = h5py.File(video_folder + 'train_frames.hdf5', 'w')
# f_hdf5_train.create_dataset('train_frames', data = train_frames, compression = 'gzip', compression_opts = 9, chunks=True)
# f_hdf5_train.close()

# f_hdf5_test = h5py.File(video_folder + 'test_frames.hdf5', 'w')
# f_hdf5_test.create_dataset('test_frames', data = test_frames, compression = 'gzip', compression_opts = 9, chunks=True)
# f_hdf5_test.close()
from keras import backend as K
K.set_learning_phase(1)


# Define Model
video_input = Input(shape=(NUM_FRAMES, 224, 224, 3))
convnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
for layer in convnet_model.layers:
    layer.trainable = False

encoded_frame_sequence = TimeDistributed(convnet_model)(video_input)
encoded_video = Bidirectional(LSTM(results.lstm_size,implementation=1,dropout=0.5))(encoded_frame_sequence)
output = Dense(NUM_TAGS, activation='sigmoid')(encoded_video)

tag_model = Model(inputs=video_input, outputs=output)
tag_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
tag_model.summary()

# tag_model.load_weights('../models/lstm_vgg_'+results.tag_type+'_tag_model_augmented.h5')

# Train Model

csv_logger = CSVLogger('../logs/lstm_vgg_'+results.tag_type+'_tag_model_bidirectional.log')
checkpointer = ModelCheckpoint(filepath='../models/lstm_vgg_'+results.tag_type+'_tag_model_bidirectional.h5', verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.0001)

tag_model.fit(augmented_train_frames, augmented_train_tags, epochs=10, batch_size=16, validation_split=0.2, callbacks=[csv_logger, checkpointer, reduce_lr])


# Measure Model Performance on test set: Hamming Loss, Recall, Precision
tag_model.load_weights('../models/lstm_vgg_'+results.tag_type+'_tag_model_bidirectional.h5')
test_preds = tag_model.predict(test_frames, batch_size=16)
flipped_test_preds = tag_model.predict(flipped_test_frames, batch_size=16)
augmented_test_preds = (flipped_test_preds + test_preds)/2

train_preds = tag_model.predict(train_frames, batch_size=16)
flipped_train_preds = tag_model.predict(flipped_train_frames, batch_size=16)
augmented_train_preds = (flipped_train_preds + train_preds)/2

train_pred_vectors = {}
for idx,video in enumerate(train_list):
        train_pred_vectors[video] = train_preds[idx]


test_pred_vectors = {}
for idx,video in enumerate(test_list):
        test_pred_vectors[video] = test_preds[idx]

pred_vectors_all = {}
pred_vectors_all.update(train_pred_vectors)
pred_vectors_all.update(test_pred_vectors)
pickle.dump(pred_vectors_all, open(results.tag_type + "_vectors_predicted_bidirectional.p","wb"))

augmented_train_pred_vectors = {}
for idx,video in enumerate(train_list):
        augmented_train_pred_vectors[video] = augmented_train_preds[idx]


augmented_test_pred_vectors = {}
for idx,video in enumerate(test_list):
        augmented_test_pred_vectors[video] = augmented_test_preds[idx]

augmented_pred_vectors_all = {}
augmented_pred_vectors_all.update(augmented_train_pred_vectors)
augmented_pred_vectors_all.update(augmented_test_pred_vectors)
pickle.dump(augmented_pred_vectors_all, open(results.tag_type + "_vectors_predicted_bidirectional_augmented.p","wb"))


from sklearn.metrics import hamming_loss
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt

test_preds_binarized = copy.deepcopy(test_preds)
test_preds_binarized[test_preds>=0.5] = 1
test_preds_binarized[test_preds<0.5] = 0

print("Hamming Loss: ", hamming_loss(test_tags, test_preds_binarized))

augmented_test_preds_binarized = copy.deepcopy(augmented_test_preds)
augmented_test_preds_binarized[augmented_test_preds>=0.5] = 1
augmented_test_preds_binarized[augmented_test_preds<0.5] = 0

print("Hamming Loss For Augmented Preds: ", hamming_loss(test_tags, augmented_test_preds_binarized))

# Calculate Micro Averaged Precision
# For each class
precision = dict()
recall = dict()
average_precision = dict()
# for i in range(NUM_TAGS):
#     precision[i], recall[i], _ = precision_recall_curve(test_tags[:, i], test_preds[:, i])
#     average_precision[i] = average_precision_score(test_tags[:, i], test_preds[:, i])

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(test_tags.ravel(), test_preds.ravel())
average_precision["micro"] = average_precision_score(test_tags, test_preds, average="micro")
print('Average precision score, micro-averaged over all classes:', average_precision["micro"])

# Plot uAP v Recall curve
plt.switch_backend("agg")
plt.figure()
plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2, where='post')
plt.fill_between(recall["micro"], precision["micro"], step='post', alpha=0.2, color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Average precision score, micro-averaged over all classes: AUC={0:0.2f}'.format(average_precision["micro"]))
plt.savefig('PR_Curve_'+results.tag_type+'_bidirectional.png')


precision = dict()
recall = dict()
average_precision = dict()
# for i in range(NUM_TAGS):
#     precision[i], recall[i], _ = precision_recall_curve(test_tags[:, i], test_preds[:, i])
#     average_precision[i] = average_precision_score(test_tags[:, i], test_preds[:, i])

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(test_tags.ravel(), augmented_test_preds.ravel())
average_precision["micro"] = average_precision_score(test_tags, augmented_test_preds, average="micro")
print('Average precision score, micro-averaged over all classes:', average_precision["micro"])

# Plot uAP v Recall curve
plt.switch_backend("agg")
plt.figure()
plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2, where='post')
plt.fill_between(recall["micro"], precision["micro"], step='post', alpha=0.2, color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Average precision score, micro-averaged over all classes: AUC={0:0.2f}'.format(average_precision["micro"]))
plt.savefig('PR_Curve_'+results.tag_type+'_bidirectional_augmented.png')
