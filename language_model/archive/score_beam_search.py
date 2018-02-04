import pickle
import numpy as np
import copy

folder = "../Youtube2Text/youtubeclips-dataset/"


fname = folder + "test.txt"
with open(fname) as f:
    content = f.readlines()

test = [x.strip() for x in content]

# Load single frame feature vectors and attribute/entity/action vectors
video_entity_vectors = pickle.load(open("../entity_classifier/entity_vectors_long.pickle", "rb"))
video_action_vectors = pickle.load(open("../action_classifier/action_vectors_long.pickle", "rb"))
video_attribute_vectors = pickle.load(open("../attribute_classifier/attribute_vectors_long.pickle", "rb"))
video_frame_features = pickle.load(open("../frame_features.pickle", "rb"))

# Remove videos for which clean captions aren't available
available_vids = set(video_entity_vectors.keys()).intersection(set(video_action_vectors.keys()).intersection(set(video_attribute_vectors.keys()).intersection(set(video_frame_features.keys()))))
test = list(set(test).intersection(available_vids))

# Read feature sizes from data 
NUM_ENTITIES   = video_entity_vectors[video_entity_vectors.keys()[0]].shape[0]
NUM_ACTIONS    = video_action_vectors[video_action_vectors.keys()[0]].shape[0]
NUM_ATTRIBUTES = video_attribute_vectors[video_attribute_vectors.keys()[0]].shape[0]
NUM_FEATURES   = video_frame_features[video_frame_features.keys()[0]].shape[1]

X_ent_test = []
X_act_test = []
X_att_test = []
X_vgg_test = []
X_prev_words_begin = []

vocabulary = pickle.load(open("vocabulary_30.p", "rb"))

# Turn vocabulary into list of words
vocabulary_words = [x[1] for x in vocabulary]

#Load the model with pre-trained weights
TRUNCATED_CAPTION_LEN = 15 + 2 
PREV_WORDS_LENGTH = TRUNCATED_CAPTION_LEN - 1
EMBEDDING_DIM = 256
NUM_WORDS = len(vocabulary_words)

# Load the video features
for video in test:
    X_ent_test.append(np.array(video_entity_vectors[video]))
    X_act_test.append(np.array(video_action_vectors[video]))
    X_att_test.append(np.array(video_attribute_vectors[video]))
    X_vgg_test.append(np.array(video_frame_features[video][0]))
    X_prev_words_begin.append([vocabulary_words.index("<bos>")] + [0]*(PREV_WORDS_LENGTH - 1))


X_ent_test  = np.array(X_ent_test)
X_act_test  = np.array(X_act_test)
X_att_test  = np.array(X_att_test)
X_vgg_test  = np.array(X_vgg_test)
X_prev_words_begin  = np.array(X_prev_words_begin)

# import keras
import keras # For keras.layers.concatenate
from keras.layers import Input, Dense, RepeatVector, Embedding
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.models import Model


# Language Model
previous_words_input = Input(shape=(PREV_WORDS_LENGTH,), dtype='float32')
previous_words_embedded = Embedding(output_dim=EMBEDDING_DIM, input_dim=NUM_WORDS, mask_zero=True, dtype='float32')(previous_words_input)
lstm1_output = LSTM(256, return_sequences=True, implementation=2, dropout=0.5, recurrent_dropout=0.5)(previous_words_embedded)
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
lstm_output = LSTM(512, return_sequences=False, implementation=2, dropout=0.5, recurrent_dropout=0.5)(lang_model_input) # fast_gpu implementation
generated_word = Dense(NUM_WORDS, activation='softmax')(lstm_output)

beam_model = Model(inputs=[entities_input, actions_input, attributes_input, single_frame_input, previous_words_input], outputs=generated_word)
beam_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

beam_model.summary()

beam_model.load_weights("../models/beam_model_basic_maxvalacc_smallvoc_longtags_05.h5")

#preds = beam_model.predict([X_ent_test, X_act_test, X_att_test, X_vgg_test, X_prev_words_begin])

# helper function to sample an index from a probability array
def sample(preds, temperature=1.0):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

# # Load All Captions
# fname = folder + "matched_all_descriptions.csv"
# with open(fname) as f:
#     content = f.readlines()

# all_captions = [(x.strip().split(",")[0],x.strip().split(",")[1]) for x in content]

# # Write correct caption file
# correct_captions = open("scoring_results/correct_captions_ref.sgm","w")
# print >>correct_captions, '<refset setid="y2txt" srclang="any" trglang="en">'
# print >>correct_captions, '<doc sysid="ref" docid="y2txt" genre="vidcap" origlang="en">'
# print >>correct_captions, '<p>'
# for video,caption in all_captions:
#     if video in test:
#         correct_sentence = []
#         print >>correct_captions, '<seg id="'+str(test.index(video))+'">' + caption + '</seg>'
# print >>correct_captions, '</p>'
# print >>correct_captions, '</doc>'
# print >>correct_captions, '</refset>'
# correct_captions.close()

def indices(k):
    combos = []
    for x in range(k):
        for y in range(k):
            combos.append((x,y))
    
    return combos


def greedy_search(captioning_model, prev_words_input, other_inputs):
    for itr in range(PREV_WORDS_LENGTH-1):
        predictions = captioning_model.predict(other_inputs + [prev_words_input])
        for idx,video in enumerate(test):
            prev_words_input[idx][itr+1] = np.argmax(predictions[idx])
    
    return prev_words_input


def beam_search(captioning_model, prev_words_input, other_inputs, k):
    top_k_predictions = [copy.deepcopy(prev_words_input) for _ in range(k)]
    top_k_score = np.array([[0.0]*top_k_predictions[0].shape[0]]*k)

    # First Iteration 
    predictions = captioning_model.predict(other_inputs + [prev_words_input])
    for idx,video in enumerate(test):
        for version in range(k):
            top_k_predictions[version][idx][1] = np.argsort(predictions[idx])[-(version+1)]
            top_k_score[version][idx] = np.sort(predictions[idx])[-(version+1)]

    for itr in range(2,PREV_WORDS_LENGTH):
        top_k_copy = copy.deepcopy(top_k_predictions)
        print top_k_predictions[0][0]
        print top_k_predictions[1][0]
        print top_k_predictions[2][0]
        predictions = [captioning_model.predict(other_inputs + [top_k_predictions[version]])  for version in range(k)]
        for idx,video in enumerate(test):
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

preds, scores = beam_search(beam_model, X_prev_words_begin, [X_ent_test, X_act_test, X_att_test, X_vgg_test], 3)

print len(preds), "x", preds[0].shape
print len(scores),"x", scores[0].shape

preds = preds[-1]

print scores[:,0]
#scores = scores[-1]


# replace <unk>, <BOS>, <EOS> with nothing
vocabulary[0]  = (0,"")
vocabulary[-1] = (0,"")
vocabulary[-2] = (0,"")
vocabulary[-3] = (0,"")

beam_captions = open("scoring_results/beam_search_smallvoc_maxacc.sgm","w")
print >>beam_captions, '<tstset trglang="en" setid="y2txt" srclang="any">'
print >>beam_captions, '<doc sysid="langmodel" docid="y2txt" genre="vidcap" origlang="en">'
print >>beam_captions, '<p>'
for idx,video in enumerate(test):
        sentence = []
        for word in preds[idx]:
            sentence.append(vocabulary[word][1])
        print >>beam_captions, '<seg id="'+str(test.index(video))+'">' + (" ".join(sentence)).strip() + '</seg>'
print >>beam_captions, '</p>'
print >>beam_captions, '</doc>'
print >>beam_captions, '</tstset>'
beam_captions.close()

preds = greedy_search(beam_model, X_prev_words_begin, [X_ent_test, X_act_test, X_att_test, X_vgg_test])

greedy_captions = open("scoring_results/greedy_search_smallvoc_maxacc.sgm","w")
print >>greedy_captions, '<tstset trglang="en" setid="y2txt" srclang="any">'
print >>greedy_captions, '<doc sysid="langmodel" docid="y2txt" genre="vidcap" origlang="en">'
print >>greedy_captions, '<p>'
for idx,video in enumerate(test):
        greedy_sentence = []
        for word in preds[idx]:
            greedy_sentence.append(vocabulary[word][1])
        print >>greedy_captions, '<seg id="'+str(test.index(video))+'">' + (" ".join(greedy_sentence)).strip() + '</seg>'
print >>greedy_captions, '</p>'
print >>greedy_captions, '</doc>'
print >>greedy_captions, '</tstset>'
greedy_captions.close()

# hot_captions = open("scoring_results/hot_captions_batched.sgm","w")
# print >>hot_captions, '<tstset trglang="en" setid="y2txt" srclang="any">'
# print >>hot_captions, '<doc sysid="langmodel" docid="y2txt" genre="vidcap" origlang="en">'
# print >>hot_captions, '<p>'
# for idx,video in enumerate(test):
#         hot_sentence = []
#         for word in preds[idx]:
#             hot_sentence.append(vocabulary[sample(word, 0.1)][1])
#         print >>hot_captions, '<seg id="'+str(test.index(video))+'">' + " ".join(hot_sentence) + '</seg>'
# print >>hot_captions, '</p>'
# print >>hot_captions, '</doc>'
# print >>hot_captions, '</tstset>'
# hot_captions.close()
