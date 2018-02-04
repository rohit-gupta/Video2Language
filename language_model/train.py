from random import shuffle
import numpy as np
import pickle
import sys

folder = "../Youtube2Text/youtubeclips-dataset/"

# Load Vocabulary
fname = folder + "vocabulary.txt"
with open(fname) as f:
    content = f.readlines()

vocabulary = [(int(x.strip().split(",")[0]),x.strip().split(",")[1]) for x in content]

# Compute number of words and truncate Vocabulary
MIN_REPEAT = 10
NUM_WORDS = 0
for count, word in vocabulary:
        if count >= MIN_REPEAT:
                NUM_WORDS +=1

# Turn vocabulary into list of words and counts
vocabulary = vocabulary[:NUM_WORDS]
# weights = 1.0 + 1*MIN_REPEAT/voc_counts

# class_weights = {}

# for itr in range(1,NUM_WORDS+1):
# 	class_weights[itr] = weights[itr-1]

# # <pad> & <bos> will never occur and <eos> should get slightly lower weight
# class_weights[0] = 0.0 # <pad>
# class_weights[max(class_weights.keys())+1] = 0.0 # <bos>
# class_weights[max(class_weights.keys())+1] = 0.9 # <eos>
# class_weights[max(class_weights.keys())+1] = 1.0 # <unk>

vocabulary = [(0,"<pad>")] + vocabulary
vocabulary.append((0,"<bos>"))
vocabulary.append((0,"<eos>"))
vocabulary.append((0,"<unk>"))

vocabulary_counts = [x[0] for x in vocabulary]
vocabulary_words = [x[1] for x in vocabulary]
voc_counts = np.array(vocabulary_counts, dtype=np.float32)

with open("vocabulary_10.p", "wb") as f:
	pickle.dump(vocabulary, f)


# Load Captions
#fname = folder + "descriptions_normalized.csv"
fname = folder + "cleaned_descriptions.csv"
with open(fname) as f:
    content = f.readlines()

captions = [(x.strip().split(",")[0],x.strip().split(",")[1]) for x in content]

# calculate maximum caption length. +2 for BOS & EOS
MAX_CAPTION_LEN = max([len(x[1].split(" ")) for x in captions])
#TRUNCATED_CAPTION_LEN = int(np.percentile([len(x[1].split(" ")) + 2 for x in captions], 95.0) + 2)
TRUNCATED_CAPTION_LEN = 15 + 2
MIN_CAPTION_LEN = 4 + 2
NUM_WORDS += 4 # <BOS>, <EOS>, <unk>, <pad>

print "MAX_CAPTION_LEN: ", MAX_CAPTION_LEN
print "TRUNCATED_CAPTION_LEN: ", TRUNCATED_CAPTION_LEN
print "NUM_WORDS: ", NUM_WORDS

def one_hot_encode(sentence,lexicon):
	if len(sentence) > TRUNCATED_CAPTION_LEN:
		sentence = sentence[:TRUNCATED_CAPTION_LEN-1]
		sentence.append('<eos>')
	encoded = []
	for index,word in enumerate(sentence):
		if word not in lexicon:
			encoded.append(len(lexicon)-1)
		else:
			encoded.append(lexicon.index(word))
	return encoded

encoded_captions = []

# One-hot encoding of captions
for video, caption in captions:
	if len(caption) >= MIN_CAPTION_LEN:
		encoded_captions.append((video, one_hot_encode(['<bos>'] + caption.split(" ") + ['<eos>'],vocabulary_words)))

print encoded_captions[0]
print encoded_captions[1]
print encoded_captions[3]


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
# video_entity_vectors = pickle.load(open("../entity_classifier/entity_vectors_long.pickle", "rb"))
# video_action_vectors = pickle.load(open("../action_classifier/action_vectors_long.pickle", "rb"))
# video_attribute_vectors = pickle.load(open("../attribute_classifier/attribute_vectors_long.pickle", "rb"))

video_entity_vectors = pickle.load(open("../advanced_tag_models/entity_simple_predicted_tags.pickle", "rb"))
video_action_vectors = pickle.load(open("../advanced_tag_models/action_simple_predicted_tags.pickle", "rb"))
video_attribute_vectors = pickle.load(open("../advanced_tag_models/attribute_simple_predicted_tags.pickle", "rb"))

# video_frame_features = pickle.load(open("../frame_features.pickle", "rb"))
video_frame_features = pickle.load(open("../frame_features/average_frame_features.pickle", "rb"))

# Remove videos for which clean captions aren't available
# available_vids = set(video_entity_vectors.keys()).intersection(set(video_action_vectors.keys()).intersection(set(video_attribute_vectors.keys()).intersection(set(video_frame_features.keys()))))
# test = set(test).intersection(available_vids)
# train = set(train).intersection(available_vids)

print str(len(train)) + " Training Videos"
print str(len(test)) + " Test Videos"


# Read feature sizes from data 
NUM_ENTITIES   = video_entity_vectors[video_entity_vectors.keys()[0]].shape[0]
NUM_ACTIONS    = video_action_vectors[video_action_vectors.keys()[0]].shape[0]
NUM_ATTRIBUTES = video_attribute_vectors[video_attribute_vectors.keys()[0]].shape[0]
NUM_FEATURES   = video_frame_features[video_frame_features.keys()[0]].shape[0] # For ResNet Features
#NUM_FEATURES   = video_frame_features[video_frame_features.keys()[0]].shape[1] # For VGG Features


X_ent_train = []
X_act_train = []
X_att_train = []
X_vgg_train = []
X_prev_words_train = []
Y_next_word_train  = []


# Pre-processing for beam_architecture
for video,caption in encoded_captions:
	if video in train:
		for idx, word in enumerate(caption[1:]):
			X_ent_train.append(np.where(video_entity_vectors.get(video,np.zeros(NUM_ENTITIES)) > 0.015, 1, 0))
			X_act_train.append(np.where(video_action_vectors.get(video,np.zeros(NUM_ACTIONS)) > 0.015, 1, 0))
			X_att_train.append(np.where(video_attribute_vectors.get(video,np.zeros(NUM_ATTRIBUTES)) > 0.015, 1, 0))
			#X_vgg_train.append(video_frame_features[video][0]) # For VGG Single Frame Features
			X_vgg_train.append(video_frame_features[video]) # For ResNet Avg Frame Features
			X_prev_words_train.append(caption[0:idx+1])
			Y_next_word_train.append(word)

for itr in range(TRUNCATED_CAPTION_LEN):
	print X_prev_words_train[itr]
	print Y_next_word_train[itr]


# Padding
for idx,prev_words in enumerate(X_prev_words_train):
	for i in range(len(prev_words), TRUNCATED_CAPTION_LEN - 1):
		prev_words.append(0)
	temp_word_id = Y_next_word_train[idx]
	Y_next_word_train[idx] = np.zeros(NUM_WORDS, dtype=np.float32)
	Y_next_word_train[idx][temp_word_id] = 1.0

for itr in range(TRUNCATED_CAPTION_LEN):
	print X_prev_words_train[itr]
	print Y_next_word_train[itr]

X_ent_train = np.array(X_ent_train)
X_act_train = np.array(X_act_train)
X_att_train = np.array(X_att_train)
X_vgg_train = np.array(X_vgg_train)
X_prev_words_train = np.array(X_prev_words_train)
Y_next_word_train  = np.array(Y_next_word_train)


print X_ent_train.shape, X_act_train.shape, X_att_train.shape, X_vgg_train.shape
print X_prev_words_train.shape, Y_next_word_train.shape

# MAX_CAPTION_LENGTH = Y_train.shape[1]
# NUM_WORDS = Y_train.shape[2]

PREV_WORDS_LENGTH = TRUNCATED_CAPTION_LEN - 1
EMBEDDING_DIM = 256


# import keras
import keras # For keras.layers.concatenate
from keras.layers import Input, Dense, RepeatVector, Embedding
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard
from keras.optimizers import RMSprop


# Language Model
previous_words_input = Input(shape=(PREV_WORDS_LENGTH,), dtype='float32')
previous_words_embedded = Embedding(output_dim=EMBEDDING_DIM, input_dim=NUM_WORDS, mask_zero=True, dtype='float32', name="word_embedding")(previous_words_input)
lstm1_output = LSTM(256, return_sequences=True, implementation=2, dropout=0.1, recurrent_dropout=0.1)(previous_words_embedded)
words_embedding_space = TimeDistributed(Dense(EMBEDDING_DIM))(lstm1_output)

# Features Model
single_frame_input = Input(shape=(NUM_FEATURES,),   dtype='float32')
features_embedding_space_single = Dense(EMBEDDING_DIM, activation='relu', dtype='float32')(single_frame_input)
features_embedding_space = RepeatVector(PREV_WORDS_LENGTH)(features_embedding_space_single)

# Tags Model
entities_input     = Input(shape=(NUM_ENTITIES,),   dtype='float32')
actions_input      = Input(shape=(NUM_ACTIONS,),    dtype='float32')
attributes_input   = Input(shape=(NUM_ATTRIBUTES,), dtype='float32')
tags_merged = keras.layers.concatenate([entities_input, actions_input, attributes_input])
tags_embedding_space = RepeatVector(PREV_WORDS_LENGTH)(tags_merged)


# Word Generation Model
lang_model_input = keras.layers.concatenate([words_embedding_space, features_embedding_space, tags_embedding_space])
lstm_output = LSTM(512, return_sequences=False, implementation=2, dropout=0.1, recurrent_dropout=0.1)(lang_model_input) # fast_gpu implementation
generated_word = Dense(NUM_WORDS, activation='softmax')(lstm_output)

beam_model = Model(inputs=[entities_input, actions_input, attributes_input, single_frame_input, previous_words_input], outputs=generated_word)
#rmsprop_optim = RMSprop(lr=0.001,decay=0.001)	
beam_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

beam_model.summary()

model_name = 'medvoc_simplepredtags_batch128_lowdropout_avgfeat_threshold'

# Train the model
csv_logger = CSVLogger('../logs/'+model_name+'.log')
loss_checkpointer = ModelCheckpoint(filepath='../models/'+model_name+'_minvalloss.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
acc_checkpointer = ModelCheckpoint(filepath='../models/'+model_name+'_maxvalacc.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
#TensorBoard_checkpointer = TensorBoard(log_dir='./tensorboardlogs', histogram_freq=1, batch_size=128, write_graph=True, write_grads=False, write_images=False, embeddings_freq=1, embeddings_layer_names=["word_embedding"], embeddings_metadata=folder + "word_embedding_metadata.tsv")

#epoch_steps = (int(X_vgg_train.shape[0]*0.9)/128)/10

#beam_model.load_weights("../models/beam_model_basic_maxvalacc_smallvoc_longtags_03.h5")
beam_model.fit([X_ent_train, X_act_train, X_att_train, X_vgg_train, X_prev_words_train], Y_next_word_train, epochs=10, batch_size=128, validation_split=0.10,  callbacks=[csv_logger, loss_checkpointer, acc_checkpointer])
