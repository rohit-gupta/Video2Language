import argparse
import pickle
import numpy as np

# parser = argparse.ArgumentParser()
# parser.add_argument('-t', action='store', dest='tag_type', help='(action/entity/attribute) Type of Tags to train model on')
# parser.add_argument('-s', action='store', dest='model_size', type=int, nargs='+', help='Hidden State Sizes')
# results = parser.parse_args()

print "Simple MLP Tag prediction network"

tag_types = ["entity", "action", "attribute"]
tag_names = {}
tag_counts = {"entity": 481, "action": 361, "attribute": 120}
for tag_type in tag_types:
    file_loc = "../tag_generator/" + tag_type + "_long.txt"
    with open(file_loc, "r") as f:
        contents = f.readlines()
    tags = [line.split(",")[0] for line in contents]
    tag_names[tag_type] = tags
        
        

model_size = [64, 32]

features = pickle.load(open("average_frame_features.pickle","rb"))

video_features = np.expand_dims(features[features.keys()[0]], axis=0)

NUM_FEATURES = features[features.keys()[0]].shape[0]

from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.regularizers import l1_l2



for tag_type in tag_types:
    inputs = Input(shape=(NUM_FEATURES,))
    first_hidden_layer = Dropout(0.5)(Dense(model_size[0], activation='relu')(inputs))
    hidden_layers = [first_hidden_layer]
    for idx,hidden_neurons in enumerate(model_size[1:]):
        hidden_layers.append(Dropout(0.5)(Dense(hidden_neurons, activation='relu')(hidden_layers[idx])))
    
    predictions = Dense(tag_counts[tag_type], activation='sigmoid', activity_regularizer=l1_l2(0.001,0.001))(hidden_layers[-1])
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='rmsprop',loss='binary_crossentropy', metrics=['accuracy'])
    model.load_weights("../models/"+tag_type+"_simple_tag_model_regularized.h5")
    pred_tags = model.predict_on_batch(video_features)[0]
    print [tag_names[tag_type][tag_idx] for tag_idx in np.arange(pred_tags.shape[0])[pred_tags > 0.01]]





