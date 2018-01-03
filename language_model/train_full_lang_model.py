from random import shuffle
import numpy as np
import pickle

folder = "../Youtube2Text/youtubeclips-dataset/"

# Load Vocabulary
fname = folder + "words.txt"
with open(fname) as f:
    content = f.readlines()

full_vocabulary = [(int(x.strip().split()[0]),x.strip().split()[1])
for x in content]

# Compute number of words and truncate Vocabulary
MIN_REPEAT = 3
NUM_WORDS = 0
for count, word in full_vocabulary:
        if count >= MIN_REPEAT:
                NUM_WORDS +=1

# # Add meta characters to vocabulary
# vocabulary = full_vocabulary[:NUM_WORDS]
# shuffle(vocabulary) # shuffles in-place
# vocabulary.append((0,"<BOS>"))
# vocabulary.append((0,"<EOS>"))
# vocabulary = [(0,"<unk>")] + vocabulary
# pickle.dump(vocabulary, open("vocabulary.p", "wb"))

vocabulary = pickle.load(open("vocabulary.p", "rb"))

# Turn vocabulary into list of words
vocabulary_words = [x[1] for x in vocabulary]


# Load Captions
#fname = folder + "descriptions_normalized.csv"
fname = folder + "matched_all_descriptions.csv"
with open(fname) as f:
    content = f.readlines()

captions = [(x.strip().split(",")[0],x.strip().split(",")[1]) for x in content]

# calculate maximum caption length. +2 for BOS & EOS
MAX_CAPTION_LEN = max([len(x[1].split(" ")) for x in captions])
#TRUNCATED_CAPTION_LEN = int(np.percentile([len(x[1].split(" ")) + 2 for x in captions], 95.0) + 2)
TRUNCATED_CAPTION_LEN = 10 + 2
MIN_CAPTION_LEN = 4 + 2
NUM_WORDS += 3 # <BOS>, <EOS>, <unk>

print "MAX_CAPTION_LEN: ", MAX_CAPTION_LEN
print "TRUNCATED_CAPTION_LEN: ", TRUNCATED_CAPTION_LEN
print "NUM_WORDS: ", NUM_WORDS

def one_hot_encode(sentence,lexicon):
	sentence = sentence[:TRUNCATED_CAPTION_LEN]
	encoded = []
	for index,word in enumerate(sentence):
		encoded.append(np.zeros(NUM_WORDS, dtype=np.float32))
		if word not in lexicon:
			encoded[index][0] = 1.0
		else:
			encoded[index][lexicon.index(word)] = 1.0
	for i in range(len(sentence), TRUNCATED_CAPTION_LEN):
		encoded.append(np.zeros(NUM_WORDS, dtype=np.float32))
		encoded[i][0] = 1.0
	return encoded

encoded_captions = []

# One-hot encoding of captions
for video, caption in captions:
	if len(caption) >= MIN_CAPTION_LEN:
		encoded_captions.append((video, one_hot_encode(['<BOS>'] + caption.split(" ") + ['<EOS>'],vocabulary_words)))


# Load Train/Test split
fname = folder + "train.txt"
with open(fname) as f:
    content = f.readlines()

train = [x.strip() for x in content]

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
train = set(train).intersection(available_vids)

print str(len(train)) + " Training Videos"
print str(len(test)) + " Test Videos"


# Read feature sizes from data 
NUM_ENTITIES   = video_entity_vectors[video_entity_vectors.keys()[0]].shape[0]
NUM_ACTIONS    = video_action_vectors[video_action_vectors.keys()[0]].shape[0]
NUM_ATTRIBUTES = video_attribute_vectors[video_attribute_vectors.keys()[0]].shape[0]
NUM_FEATURES   = video_frame_features[video_frame_features.keys()[0]].shape[1]


X1_train = []
X2_train = []
X3_train = []
X4_train = []
Y_train  = []

X1_test = []
X2_test = []
X3_test = []
X4_test = []
Y_test  = []

# encoded_captions = pickle.load(open("encoded_captions_len_4_10.p", "rb"))
# vocabulary = pickle.load(open("vocabulary.p", "rb"))

for video,caption in encoded_captions:
	if video in train:
		X1_train.append(np.array(video_entity_vectors[video]))
		X2_train.append(np.array(video_action_vectors[video]))
		X3_train.append(np.array(video_attribute_vectors[video]))
		X4_train.append(np.array(video_frame_features[video][0]))
		Y_train.append(caption)
	if video in test:
		X1_test.append(np.array(video_entity_vectors[video]))
		X2_test.append(np.array(video_action_vectors[video]))
		X3_test.append(np.array(video_attribute_vectors[video]))
		X4_test.append(np.array(video_frame_features[video][0]))
		Y_test.append(caption)


X1_train = np.array(X1_train)
X2_train = np.array(X2_train)
X3_train = np.array(X3_train)
X4_train = np.array(X4_train)
Y_train  = np.array(Y_train)

X1_test = np.array(X1_test)
X2_test = np.array(X2_test)
X3_test = np.array(X3_test)
X4_test = np.array(X4_test)
Y_test  = np.array(Y_test)


MAX_CAPTION_LENGTH = Y_train.shape[1]
NUM_WORDS = Y_train.shape[2]


import keras
from keras.layers import Input, Dense, RepeatVector
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.callbacks import CSVLogger, ModelCheckpoint

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
# lstm_output = LSTM(512, return_sequences=True, implementation=2, dropout=0.5, recurrent_dropout=0.5)(lstm_input) # fast_gpu implementation
output = TimeDistributed(Dense(NUM_WORDS, activation='softmax'))(lstm3_output)


lang_model = Model(inputs=[entities_input, actions_input, attributes_input, single_frame_input], outputs=output)
lang_model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

lang_model.summary()

csv_logger = CSVLogger('../logs/language_model_all_captions_3layer.log')
checkpointer = ModelCheckpoint(filepath='../models/language_model_all_captions_best_weights_3layer.h5', verbose=1, save_best_only=True)

lang_model.fit([X1_train, X2_train, X3_train, X4_train], Y_train, epochs=20, batch_size=100, validation_split=0.1, callbacks=[csv_logger, checkpointer])
lang_model.save('../models/language_model_all_captions_3layer.h5')

preds = lang_model.predict([X1_test, X2_test, X3_test, X4_test])

# helper function to sample an index from a probability array
def sample(preds, temperature=1.0):
	preds = np.asarray(preds).astype('float64')
	preds = np.log(preds) / temperature
	exp_preds = np.exp(preds)
	preds = exp_preds / np.sum(exp_preds)
	probas = np.random.multinomial(1, preds, 1)
	return np.argmax(probas)

for idx in range(5):
	print "".join(['*~']*40)
	greedy_sentence = []
	hot_sentence = []
	correct_sentence = []
	for word in preds[idx]:
		greedy_sentence.append(vocabulary[np.argmax(word)][1])
		hot_sentence.append(vocabulary[sample(word)][1])
		correct_sentence.append(vocabulary[np.argmax(word)][1])
	print " ".join(correct_sentence)
	print " ".join(greedy_sentence)
	print " ".join(hot_sentence)