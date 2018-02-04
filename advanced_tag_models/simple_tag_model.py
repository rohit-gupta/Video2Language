import argparse
import pickle
import numpy as np

video_folder = "../Youtube2Text/youtubeclips-dataset/"

# Load Train/Test split
fname = video_folder + "train.txt"
with open(fname) as f:
    content = f.readlines()

train_list = [x.strip() for x in content]

fname = video_folder + "test.txt"
with open(fname) as f:
    content = f.readlines()

test_list = [x.strip() for x in content]


parser = argparse.ArgumentParser()
parser.add_argument('-t', action='store', dest='tag_type', help='(action/entity/attribute) Type of Tags to train model on')
parser.add_argument('-s', action='store', dest='model_size', type=int, nargs='+', help='Hidden State Sizes')
results = parser.parse_args()


tag_vectors = pickle.load(open("../"+results.tag_type+"_classifier/" + results.tag_type + "_vectors_long.pickle", "rb"))
# entity_vectors = pickle.load(open("../entity_classifier/entity_vectors_long.pickle", "rb"))
# attribute_vectors = pickle.load(open("../attribute_classifier/attribute_vectors_long.pickle", "rb"))

NUM_TAGS = tag_vectors[tag_vectors.keys()[0]].shape[0]

train_tags = np.array([tag_vectors.get(video,np.zeros(NUM_TAGS)) for video in train_list])
test_tags  = np.array([tag_vectors.get(video,np.zeros(NUM_TAGS)) for video in test_list])


features = pickle.load(open("../frame_features/average_frame_features.pickle","rb"))
NUM_FEATURES = features[features.keys()[0]].shape[0]

train_features = np.array([features.get(video,np.zeros(NUM_TAGS)) for video in train_list])
test_features  = np.array([features.get(video,np.zeros(NUM_TAGS)) for video in test_list])




from keras.layers import Input, Dense,Dropout
from keras.models import Model
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras import regularizers

inputs = Input(shape=(NUM_FEATURES,))
first_hidden_layer = Dropout(0.5)(Dense(results.model_size[0], activation='relu')(inputs))
hidden_layers = [first_hidden_layer]
for idx,hidden_neurons in enumerate(results.model_size[1:]):
        hidden_layers.append(Dropout(0.5)(Dense(hidden_neurons, activation='relu')(hidden_layers[idx])))

predictions = Dense(NUM_TAGS, activation='sigmoid', activity_regularizer=regularizers.l1_l2(0.001,0.001))(hidden_layers[-1])
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='rmsprop',loss='binary_crossentropy', metrics=['accuracy'])


csv_logger = CSVLogger('../logs/'+results.tag_type+'_simple_tag_model_regularized.log')
checkpointer = ModelCheckpoint(filepath='../models/'+results.tag_type+'_simple_tag_model_regularized.h5', verbose=1, save_best_only=True)


model.fit(train_features, train_tags, batch_size=32,epochs=100,validation_split=0.1,shuffle=True, callbacks=[csv_logger,checkpointer])  # starts training

model.load_weights("../models/"+results.tag_type+"_simple_tag_model_regularized.h5")

pred_tags = model.predict(test_features)
pred_train_tags = model.predict(train_features)



pred_dict = {}
for idx,video in enumerate(train_list):
        pred_dict[video] = pred_train_tags[idx]

for idx,video in enumerate(test_list):
        pred_dict[video] = pred_tags[idx]


pickle.dump(pred_dict, open(results.tag_type+"_simple_predicted_tags.pickle","wb"))


actual_tags = test_tags

from sklearn.metrics import hamming_loss
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt


# Calculate Micro Averaged Precision
# For each class
precision = dict()
recall = dict()
average_precision = dict()
# for i in range(NUM_TAGS):
#     precision[i], recall[i], _ = precision_recall_curve(test_tags[:, i], test_preds[:, i])
#     average_precision[i] = average_precision_score(test_tags[:, i], test_preds[:, i])

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(actual_tags.ravel(), pred_tags.ravel())
average_precision["micro"] = average_precision_score(actual_tags, pred_tags, average="micro")
print 'Average precision score, micro-averaged over all classes:', average_precision["micro"]

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
plt.savefig('PR_Curve_simple_model_regularized_'+results.tag_type+'.png')

