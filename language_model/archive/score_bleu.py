import pickle
import numpy as np
from __future__ import print_function

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
test = list(set(test).intersection(available_vids))

# Read feature sizes from data
NUM_ENTITIES   = video_entity_vectors[video_entity_vectors.keys()[0]].shape[0]
NUM_ACTIONS    = video_action_vectors[video_action_vectors.keys()[0]].shape[0]
NUM_ATTRIBUTES = video_attribute_vectors[video_attribute_vectors.keys()[0]].shape[0]
NUM_FEATURES   = video_frame_features[video_frame_features.keys()[0]].shape[1]

X1_test = []
X2_test = []
X3_test = []
X4_test = []

vocabulary = pickle.load(open("vocabulary.p", "rb"))

# replace <unk>, <BOS>, <EOS> with nothing
vocabulary[0]  = (0,"")
vocabulary[-1] = (0,"")
vocabulary[-2] = (0,"")


# Load the video features
for video in test:
    X1_test.append(np.array(video_entity_vectors[video]))
    X2_test.append(np.array(video_action_vectors[video]))
    X3_test.append(np.array(video_attribute_vectors[video]))
    X4_test.append(np.array(video_frame_features[video][0]))


X1_test = np.array(X1_test)
X2_test = np.array(X2_test)
X3_test = np.array(X3_test)
X4_test = np.array(X4_test)


#Load the model with pre-trained weights
# MAX_CAPTION_LENGTH = len(encoded_captions[0][1])
# NUM_WORDS = len(encoded_captions[0][1][0])

MAX_CAPTION_LENGTH = 12
NUM_WORDS = len(vocabulary)

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
lstm1_output = LSTM(512, return_sequences=True, implementation=2, dropout=0.5, recurrent_dropout=0.5)(lstm_input) # fast_gpu implementation
lstm2_output = LSTM(512, return_sequences=True, implementation=2, dropout=0.5, recurrent_dropout=0.5)(lstm1_output) # fast_gpu implementation
lstm3_output = LSTM(512, return_sequences=True, implementation=2, dropout=0.5, recurrent_dropout=0.5)(lstm2_output) # fast_gpu implementation
output = TimeDistributed(Dense(NUM_WORDS, activation='softmax'))(lstm3_output)


lang_model = Model(inputs=[entities_input, actions_input, attributes_input, single_frame_input], outputs=output)
lang_model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

lang_model.summary()

lang_model.load_weights("../models/language_model_batched_weights_batch12.h5")

preds = lang_model.predict([X1_test, X2_test, X3_test, X4_test])

# helper function to sample an index from a probability array
def sample(preds, temperature=1.0):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

# Load All Captions
# fname = folder + "matched_all_descriptions.csv"
# with open(fname) as f:
#     content = f.readlines()

# all_captions = [(x.strip().split(",")[0],x.strip().split(",")[1]) for x in content]

# Write correct caption file
# correct_captions = open("correct_captions_all_data_finetuned.sgm","w")
# print(>>correct_captions, '<refset setid="y2txt" srclang="any" trglang="en">')
# print(>>correct_captions, '<doc sysid="ref" docid="y2txt" genre="vidcap" origlang="en">')
# print(>>correct_captions, '<p>')
# for video,caption in all_captions:
#     if video in test:
#         correct_sentence = []
#         print(>>correct_captions, '<seg id="'+str(test.index(video))+'">' + caption + '</seg>')
# print(>>correct_captions, '</p>')
# print(>>correct_captions, '</doc>')
# print(>>correct_captions, '</refset>')
# correct_captions.close()



greedy_captions = open("scoring_results/greedy_captions_batched.sgm","w")
print(>>greedy_captions, '<tstset trglang="en" setid="y2txt" srclang="any">')
print(>>greedy_captions, '<doc sysid="langmodel" docid="y2txt" genre="vidcap" origlang="en">')
print(>>greedy_captions, '<p>')
for idx,video in enumerate(test):
        greedy_sentence = []
        for word in preds[idx]:
            greedy_sentence.append(vocabulary[np.argmax(word)][1])
        print(>>greedy_captions, '<seg id="'+str(test.index(video))+'">' + " ".join(greedy_sentence) + '</seg>')
print(>>greedy_captions, '</p>')
print(>>greedy_captions, '</doc>')
print(>>greedy_captions, '</tstset>')
greedy_captions.close()

hot_captions = open("scoring_results/hot_captions_batched.sgm","w")
print(>>hot_captions, '<tstset trglang="en" setid="y2txt" srclang="any">')
print(>>hot_captions, '<doc sysid="langmodel" docid="y2txt" genre="vidcap" origlang="en">')
print(>>hot_captions, '<p>')
for idx,video in enumerate(test):
        hot_sentence = []
        for word in preds[idx]:
            hot_sentence.append(vocabulary[sample(word, 0.1)][1])
        print(>>hot_captions, '<seg id="'+str(test.index(video))+'">' + " ".join(hot_sentence) + '</seg>')
print(>>hot_captions, '</p>')
print(>>hot_captions, '</doc>')
print(>>hot_captions, '</tstset>')
hot_captions.close()
