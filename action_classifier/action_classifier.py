#import numpy as np
import pickle
import numpy as np
from __future__ import print_function

folder = "../Youtube2Text/youtubeclips-dataset/"

# Load Train/Test split
fname = folder + "train.txt"
with open(fname) as f:
    content = f.readlines()

train = [x.strip() for x in content]

fname = folder + "test.txt"
with open(fname) as f:
    content = f.readlines()

test = [x.strip() for x in content]

# Load single frame feature vectors and action vectors
video_action_vectors = pickle.load(open("action_vectors.pickle", "rb"))
video_frame_features = pickle.load(open("../frame_features.pickle", "rb"))

# Remove videos for which clean captions aren't available
available_vids = set(video_action_vectors.keys()).intersection(set(video_frame_features.keys()))
train = set(train).intersection(available_vids)
test = set(test).intersection(available_vids)

# Split train and test set
X_train = np.array([video_frame_features[video][0] for video in train])
Y_train = np.array([video_action_vectors[video] for video in train])
X_test = np.array([video_frame_features[video][0] for video in test])
Y_test = np.array([video_action_vectors[video] for video in test])

# Dimensions
num_actions = video_action_vectors[video_action_vectors.keys()[0]].shape[0]
feature_size = video_frame_features[video_frame_features.keys()[0]].shape[1]

print("Training action classifier model ...")

# Define the action classifier model
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import CSVLogger, ModelCheckpoint

action_model = Sequential()
action_model.add(Dense(128, activation='relu', input_dim=feature_size))
action_model.add(Dropout(0.5))
action_model.add(Dense(64, activation='relu'))
action_model.add(Dropout(0.5))
action_model.add(Dense(num_actions, activation='sigmoid'))

action_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Save a visualization of the model and print summary
#from keras.utils import plot_model
#plot_model(action_model, to_file='action_classifier_model.png')
action_model.summary()

# Train the model
csv_logger = CSVLogger('../logs/action_classifier_training_dropout_0p5.log')
checkpointer = ModelCheckpoint(filepath='../models/action_classifier_best_weights_dropout_0p5.h5', verbose=1, save_best_only=True)
action_model.fit(X_train, Y_train, epochs=100, batch_size=100, validation_split=0.2, callbacks=[csv_logger, checkpointer])

action_model.save('../models/action_classifier_model_dropout_0p5.h5')

preds = action_model.predict(X_test)


from sklearn.metrics import hamming_loss

preds_binarized = preds
preds_binarized[preds>=0.5] = 1
preds_binarized[preds<0.5] = 0

print("Hamming Loss: ", hamming_loss(Y_test, preds_binarized))
