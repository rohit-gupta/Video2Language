import argparse
import pickle
import numpy as np
import copy
from __future__ import print_function

# parser = argparse.ArgumentParser()
# parser.add_argument('-t', action='store', dest='tag_type', help='(action/entity/attribute) Type of Tags to train model on')
# parser.add_argument('-s', action='store', dest='model_size', type=int, nargs='+', help='Hidden State Sizes')
# results = parser.parse_args()

print("Simple MLP Tag prediction network")

tag_types = ["entity", "action", "attribute"]
tag_names = {}
tag_counts = {"entity": 481, "action": 361, "attribute": 120}
for tag_type in tag_types:
    file_loc = "../tag_generator/" + tag_type + "_long.txt"
    with open(file_loc, "r") as f:
        contents = f.readlines()
    tags = [line.split(",")[0] for line in contents]
    tag_names[tag_type] = tags


print("Predicting tags ...")

model_size = [64, 32] # These are the dimensions for the pre-trained model, change them if necessary

features = pickle.load(open("average_frame_features.pickle","rb"))

video_features = np.expand_dims(features[features.keys()[0]], axis=0)

NUM_FEATURES = features[features.keys()[0]].shape[0]
THRESHOLD = 0.01

from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.regularizers import l1_l2

tags_for_langmodel = {}

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
    pred_tags = model.predict_on_batch(video_features)
    pred_tags[0, pred_tags[0] > THRESHOLD] = 1
    pred_tags[0, pred_tags[0] < THRESHOLD] = 0
    tags_for_langmodel[tag_type] = pred_tags
    # print([tag_names[tag_type][tag_idx] for tag_idx in np.arange(pred_tags.shape[0])[pred_tags > 0.01]])


from language_model import get_language_model

vocabulary = pickle.load(open("../language_model/vocabulary_10.p", "rb"))
vocabulary_words = [x[1] for x in vocabulary]

VOCABULARY_SIZE = len(vocabulary_words)

NUM_ENTITIES   = tags_for_langmodel['entity'].shape[1]
NUM_ACTIONS    = tags_for_langmodel['action'].shape[1]
NUM_ATTRIBUTES = tags_for_langmodel['attribute'].shape[1]

LSTM_SIZE = 512     # These are the dimensions for the pre-trained model, change them if necessary
EMBEDDING_DIM = 256 # These are the dimensions for the pre-trained model, change them if necessary
NUM_PREV_WORDS = 15 + 2 -1 # Total Words in Caption + <bos> + <eos> - active word

beam_model = get_language_model(NUM_PREV_WORDS, VOCABULARY_SIZE, EMBEDDING_DIM, NUM_FEATURES, NUM_ENTITIES, NUM_ACTIONS, NUM_ATTRIBUTES, LSTM_SIZE, 0.0, 0.0) # Dropout is inactive during inference

prev_words_begin = []
prev_words_begin.append([vocabulary_words.index("<bos>")] + [0]*(NUM_PREV_WORDS - 1))
prev_words_begin = np.array(prev_words_begin)

print("Loading language model ...")

beam_model.load_weights("../models/medvoc_simplepredtags_batch128_lowdropout_avgfeat_threshold_maxvalacc.h5")


def indices(k):
    combos = []
    for x in range(k):
        for y in range(k):
            combos.append((x,y))

    return combos

def beam_search(captioning_model, prev_words_input, other_inputs, k):
    top_k_predictions = [copy.deepcopy(prev_words_input) for _ in range(k)]
    top_k_score = np.array([[0.0]*top_k_predictions[0].shape[0]]*k)

    # First Iteration
    predictions = captioning_model.predict(other_inputs + [prev_words_input])
    for idx in range(prev_words_input.shape[0]):
        for version in range(k):
            top_k_predictions[version][idx][1] = np.argsort(predictions[idx])[-(version+1)]
            top_k_score[version][idx] = np.sort(predictions[idx])[-(version+1)]

    for itr in range(2,NUM_PREV_WORDS):
        top_k_copy = copy.deepcopy(top_k_predictions)
        print("Running beam search iteration number #", itr, "...")
#         print(top_k_predictions[0][0])
#         print(top_k_predictions[1][0])
#         print(top_k_predictions[2][0])
#         print(top_k_predictions[3][0])
#         print(top_k_predictions[4][0])
        predictions = [captioning_model.predict(other_inputs + [top_k_predictions[version]])  for version in range(k)]
        for idx in range(prev_words_input.shape[0]):
            scores = []
            for version,lookahead in indices(k):
                scores.append(np.sort(predictions[version][idx])[-(lookahead+1)]*top_k_score[version][idx])
            scores = np.array(scores)
            top_score_indices = np.argsort(scores)[-k:]
            for num, top_id in enumerate(top_score_indices):
                version, lookahead = indices(k)[top_id]
                top_k_predictions[num][idx][itr] = np.argsort(predictions[version][idx])[-(lookahead+1)]
                top_k_predictions[num][idx][:itr] = top_k_copy[version][idx][:itr]
                top_k_score[num][idx] = scores[top_id]

    return top_k_predictions, top_k_score

preds, scores = beam_search(beam_model, prev_words_begin, [tags_for_langmodel['entity'], tags_for_langmodel['action'], tags_for_langmodel['attribute'], video_features], 20)


for idx, pred in enumerate(preds):
    print(" ".join([vocabulary_words[word_idx] for word_idx in pred[0].tolist()]).replace("<eos>", "").replace("<bos>", ""), round(scores[idx][0]/np.max(scores),3))
