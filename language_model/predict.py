import pickle
import numpy as np

folder = "../Youtube2Text/youtubeclips-dataset/"


fname = folder + "test.txt"
with open(fname) as f:
    content = f.readlines()

test = [x.strip() for x in content]

# Load single frame feature vectors and attribute/entity/action vectors
video_entity_vectors = pickle.load(open("../entity_classifier/entity_vectors.pickle", "rb"))
video_action_vectors = pickle.load(open("../action_classifier/action_vectors.pickle", "rb"))
video_attribute_vectors = pickle.load(open("../attribute_classifier/attribute_vectors.pickle", "rb"))
video_frame_features = pickle.load(open("../frame_features.pickle", "rb"))

# Remove videos for which clean captions aren't available
available_vids = set(video_entity_vectors.keys()).intersection(set(video_action_vectors.keys()).intersection(set(video_attribute_vectors.keys()).intersection(set(video_frame_features.keys()))))
test = set(test).intersection(available_vids)

# Read feature sizes from data 
NUM_ENTITIES   = video_entity_vectors[video_entity_vectors.keys()[0]].shape[0]
NUM_ACTIONS    = video_action_vectors[video_action_vectors.keys()[0]].shape[0]
NUM_ATTRIBUTES = video_attribute_vectors[video_attribute_vectors.keys()[0]].shape[0]
NUM_FEATURES   = video_frame_features[video_frame_features.keys()[0]].shape[1]

X1_test = []
X2_test = []
X3_test = []
X4_test = []
Y_test  = []

encoded_captions = pickle.load(open("encoded_captions.p", "rb"))
vocabulary = pickle.load(open("vocabulary.p", "rb"))

for video,caption in encoded_captions:
        if video in test:
                X1_test.append(np.array(video_entity_vectors[video]))
                X2_test.append(np.array(video_action_vectors[video]))
                X3_test.append(np.array(video_attribute_vectors[video]))
                X4_test.append(np.array(video_frame_features[video][0]))
                Y_test.append(caption)

X1_test = np.array(X1_test)
X2_test = np.array(X2_test)
X3_test = np.array(X3_test)
X4_test = np.array(X4_test)
Y_test  = np.array(Y_test)



MAX_CAPTION_LENGTH = Y_test.shape[1]
NUM_WORDS = Y_test.shape[2]


import keras
from keras.layers import Input, Dense, RepeatVector
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.models import Model

# Inputs
entities_input     = Input(shape=(NUM_ENTITIES,),   dtype='float32')
actions_input      = Input(shape=(NUM_ACTIONS,),    dtype='float32')
attributes_input   = Input(shape=(NUM_ATTRIBUTES,), dtype='float32')
single_frame_input = Input(shape=(NUM_FEATURES,),   dtype='float32')
merged = keras.layers.concatenate([entities_input, actions_input, attributes_input, single_frame_input])

# LSTM
lstm_input = RepeatVector(MAX_CAPTION_LENGTH)(merged)
lstm_output = LSTM(256, return_sequences=True, implementation=2, dropout=0.5, recurrent_dropout=0.5)(lstm_input) # fast_gpu implementation
output = TimeDistributed(Dense(NUM_WORDS, activation='softmax'))(lstm_output)


lang_model = Model(inputs=[entities_input, actions_input, attributes_input, single_frame_input], outputs=output)
lang_model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

lang_model.summary()

lang_model.load_weights("../models/language_model_best_weights_dropout_0p5.h5")

preds = lang_model.predict([X1_test, X2_test, X3_test, X4_test])

# helper function to sample an index from a probability array
def sample(preds, temperature=1.0):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

greedy_captions = open("greedy_captions.txt","w")
hot_captions = open("hot_captions.txt","w")
correct_captions = open("correct_captions.txt","w")

for prediction in preds:
    greedy_sentence = []
    hot_sentence = []
    for word in prediction:
            hot_sentence.append(vocabulary[sample(word)][1])
            greedy_sentence.append(vocabulary[np.argmax(word)][1])
    print >>greedy_captions,  " ".join(greedy_sentence)
    print >>hot_captions,     " ".join(hot_sentence)



for sentence in Y_test:
    correct_sentence = []
    for word in sentence:
        correct_sentence.append(vocabulary[np.argmax(word)][1])
    print >>correct_captions, " ".join(correct_sentence)

greedy_captions.close()
hot_captions.close()
correct_captions.close()

