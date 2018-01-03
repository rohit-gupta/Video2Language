#import numpy as np
import pickle
import numpy as np

folder = "../Youtube2Text/youtubeclips-dataset/"

# Load Test split
fname = folder + "test.txt"
with open(fname) as f:
    content = f.readlines()

test = [x.strip() for x in content]

# Load single frame feature vectors and attribute vectors
video_attribute_vectors = pickle.load(open("attribute_vectors.pickle", "rb"))
video_frame_features = pickle.load(open("../frame_features.pickle", "rb"))

# Remove videos for which clean captions aren't available
available_vids = set(video_attribute_vectors.keys()).intersection(set(video_frame_features.keys()))
test = set(test).intersection(available_vids)

# Get test set
X_test = np.array([video_frame_features[video][0] for video in test])
Y_test = np.array([video_attribute_vectors[video] for video in test])

# Dimensions
num_attributes = video_attribute_vectors[video_attribute_vectors.keys()[0]].shape[0]
feature_size = video_frame_features[video_frame_features.keys()[0]].shape[1]

# Load the list of attributes
attributes_file = "attributes_shuffled.txt"

# read in a list of attributes
with open(folder+attributes_file) as f:
    attributes_temp = f.readlines()

attributes = [x.strip() for x in attributes_temp]


# Define the attribute classifier model
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

attribute_model = Sequential()
attribute_model.add(Dense(128, activation='relu', input_dim=feature_size))
attribute_model.add(Dropout(0.5))
attribute_model.add(Dense(64, activation='relu'))
attribute_model.add(Dropout(0.5))
attribute_model.add(Dense(num_attributes, activation='sigmoid'))

attribute_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

attribute_model.load_weights("../models/attribute_classifier_best_weights_dropout_0p5.h5")

preds = attribute_model.predict(X_test)


def attribute_vector_to_list(attribute_vec, p):
	indices = [i for i, x in enumerate(attribute_vec) if x >= p]
	return [attributes[i] for i in indices]


# print "p,  Hamming Loss,  Avg Labels, Avg Correct Labels"
# for p in np.arange(0.0, 0.6, 0.02):
# 	# Compute Hamming Loss
# 	total_predicted = 0
# 	total_correct = 0
# 	for idx,video in enumerate(test):
# 		correct_labels = set(attribute_vector_to_list(video_attribute_vectors[video],p))
# 		predicted_labels = set(attribute_vector_to_list(preds[idx],p))
# 		total_correct += len(correct_labels.intersection(predicted_labels))
# 		total_predicted += len(predicted_labels)
# 	hamming_loss = float(total_predicted - total_correct)/total_predicted
# 	avg_labels = total_predicted/float(len(test))
# 	avg_correct_labels = avg_labels*(1 - hamming_loss)
# 	print  p, hamming_loss, avg_labels, avg_correct_labels


# TODO 
# Create crude sentences for checking BLEU/METEOR
for idx,video in enumerate(test):
		predicted_labels = set(attribute_vector_to_list(preds[idx],0.3))
		print video," ".join(predicted_labels)
