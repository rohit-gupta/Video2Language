import pickle
import numpy as np
import copy
import json
import argparse
from model import get_language_model
from __future__ import print_function

folder = "../YouTube2Text/youtubeclips-dataset/"


fname = folder + "test.txt"
with open(fname) as f:
    content = f.readlines()

test = [x.strip() for x in content]

# no_clean_captions = set(['vid1690', 'vid1458', 'vid1657', 'vid1772', 'vid1515', 'vid1445', 'vid1446', 'vid1797', 'vid1855', 'vid1724', 'vid1787', 'vid1605', 'vid1455', 'vid1722', 'vid1746', 'vid1912', 'vid1301', 'vid1868', 'vid1887'])

# test = list(set(test) - no_clean_captions)




parser = argparse.ArgumentParser()
parser.add_argument('-p', action='store', dest='tag_type', help='(predicted/groundtruth) Type of Tags to use in predictions')
parser.add_argument('-t', action='store', dest='tag_threshold', type= float, help='Threshold for tag binarization')
parser.add_argument('-s', action='store', dest='lstm_size', type= int, help='Number of hidden units in LSTM model')
parser.add_argument('-m', action='store', dest='model_file', help='Model File Name')
# parser.add_argument('-d', action='store', dest='gpu', help='GPU to use')
results = parser.parse_args()

TAG_TYPE = results.tag_type
THRESHOLD = results.tag_threshold
LSTM_SIZE = results.lstm_size


# Load single frame feature vectors and attribute/entity/action vectors
if TAG_TYPE == 'predicted':
    video_entity_vectors = pickle.load(open("../advanced_tag_models/entity_simple_predicted_tags.pickle", "rb"))
    video_action_vectors = pickle.load(open("../advanced_tag_models/action_simple_predicted_tags.pickle", "rb"))
    video_attribute_vectors = pickle.load(open("../advanced_tag_models/attribute_simple_predicted_tags.pickle", "rb"))
    #video_entity_vectors = pickle.load(open("../advanced_tag_models/entity_vectors_predicted.p", "rb"))
    # video_action_vectors = pickle.load(open("../advanced_tag_models/action_vectors_predicted.p", "rb"))
    #video_attribute_vectors = pickle.load(open("../advanced_tag_models/attribute_vectors_predicted.p", "rb"))


else:
    video_entity_vectors = pickle.load(open("../entity_classifier/entity_vectors_long.pickle", "rb"))
    video_action_vectors = pickle.load(open("../action_classifier/action_vectors_long.pickle", "rb"))
    video_attribute_vectors = pickle.load(open("../attribute_classifier/attribute_vectors_long.pickle", "rb"))

video_frame_features = pickle.load(open("../frame_features/average_frame_features.pickle", "rb"))

# Remove videos for which clean captions aren't available
# available_vids = set(video_entity_vectors.keys()).intersection(set(video_action_vectors.keys()).intersection(set(video_attribute_vectors.keys()).intersection(set(video_frame_features.keys()))))
# test = list(set(test).intersection(available_vids))

# Read feature sizes from data
NUM_ENTITIES   = video_entity_vectors[video_entity_vectors.keys()[0]].shape[0]
NUM_ACTIONS    = video_action_vectors[video_action_vectors.keys()[0]].shape[0]
NUM_ATTRIBUTES = video_attribute_vectors[video_attribute_vectors.keys()[0]].shape[0]
# NUM_FEATURES   = video_frame_features[video_frame_features.keys()[0]].shape[1]
NUM_FEATURES   = video_frame_features[video_frame_features.keys()[0]].shape[0]

X_ent_test = []
X_act_test = []
X_att_test = []
X_vgg_test = []
X_prev_words_begin = []

vocabulary = pickle.load(open("vocabulary_10.p", "rb"))

# Turn vocabulary into list of words
vocabulary_words = [x[1] for x in vocabulary]

#Load the model with pre-trained weights
TRUNCATED_CAPTION_LEN = 15 + 2
NUM_PREV_WORDS = TRUNCATED_CAPTION_LEN - 1
EMBEDDING_DIM = 256
VOCABULARY_SIZE = len(vocabulary_words)

# Load the video features
for video in test:
    X_ent_test.append(np.where(np.array(video_entity_vectors.get(video, np.zeros(NUM_ENTITIES))) > THRESHOLD, 1, 0))
    X_act_test.append(np.where(np.array(video_action_vectors.get(video, np.zeros(NUM_ACTIONS))) > THRESHOLD, 1, 0))
    X_att_test.append(np.where(np.array(video_attribute_vectors.get(video, np.zeros(NUM_ATTRIBUTES))) > THRESHOLD, 1, 0))
    # X_vgg_test.append(np.array(video_frame_features[video][0]))
    X_vgg_test.append(np.array(video_frame_features[video]))
    X_prev_words_begin.append([vocabulary_words.index("<bos>")] + [0]*(NUM_PREV_WORDS - 1))


X_ent_test  = np.array(X_ent_test)
X_act_test  = np.array(X_act_test)
X_att_test  = np.array(X_att_test)
X_vgg_test  = np.array(X_vgg_test)
X_prev_words_begin  = np.array(X_prev_words_begin)


beam_model = get_language_model(NUM_PREV_WORDS, VOCABULARY_SIZE, EMBEDDING_DIM, NUM_FEATURES, NUM_ENTITIES, NUM_ACTIONS, NUM_ATTRIBUTES, LSTM_SIZE, 0.0, 0.0) # Dropout is inactive during inference

beam_model.load_weights("../models/"+results.model_file+".h5")

#preds = beam_model.predict([X_ent_test, X_act_test, X_att_test, X_vgg_test, X_prev_words_begin])

# helper function to sample an index from a probability array
def sample(preds, temperature=1.0):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

results_folder = "./"

# # Load All Captions
fname = folder + "cleaned_descriptions.csv"
with open(fname) as f:
    content = f.readlines()

all_captions = [(x.strip().split(",")[0],x.strip().split(",")[1]) for x in content]

# # Write correct caption file
correct_captions = open(results_folder + "annotations/correct_captions_ref.json","w")
correct_annotations = {}
correct_annotations['info'] = {'description': 'YouTube2Text', 'url': 'http://upplysingaoflun.ecn.purdue.edu/~yu239/datasets/youtubeclips.zip', 'version': '1.0', 'year': 2013, 'contributor': 'Guadarrama et al', u'date_created': u'2013-01-27 09:11:52.357475'}
correct_annotations['images'] = []
correct_annotations['licenses'] = [{'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/', 'id': 1, 'name': 'Attribution-NonCommercial-ShareAlike License'}]
correct_annotations['type'] = "captions"
correct_annotations['annotations'] = []

for video in test:
	correct_annotations['images'].append({'license': 1, 'url': 'https://www.youtube.com/watch?v=' + video, 'file_name': video+".avi", 'height': 360, 'width': 640, 'date_captured': u'2013-11-14 11:18:45', 'id': test.index(video)})


count = 0
for video,caption in all_captions:
	if video in test:
		correct_annotations['annotations'].append({"caption": caption, "id": count, "image_id": test.index(video), "vid_id": video})
		count +=1
correct_captions.write(json.dumps(correct_annotations, indent=4, sort_keys=True))
correct_captions.close()

def indices(k):
    combos = []
    for x in range(k):
        for y in range(k):
            combos.append((x,y))

    return combos


def greedy_search(captioning_model, prev_words_input, other_inputs):
    for itr in range(NUM_PREV_WORDS-1):
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

    for itr in range(2,NUM_PREV_WORDS):
        top_k_copy = copy.deepcopy(top_k_predictions)
        print(top_k_predictions[0][0])
        print(top_k_predictions[1][0])
        print(top_k_predictions[2][0])
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

print(len(preds), "x", preds[0].shape)
print(len(scores),"x", scores[0].shape)

preds = preds[-1]

print(scores[:,0])
#scores = scores[-1]


# replace <unk>, <BOS>, <EOS> with nothing
vocabulary[0]  = (0,"")
vocabulary[-1] = (0,"")
vocabulary[-2] = (0,"")
vocabulary[-3] = (0,"")

beam_captions_file = results_folder + "results/beam_search_"+results.model_file+results.tag_type+".json"
beam_captions = open(beam_captions_file,"w")
beam_annotations = []
for idx,video in enumerate(test):
        sentence = []
        for word in preds[idx]:
            sentence.append(vocabulary[word][1])
        beam_annotations.append({"image_id": test.index(video), "caption": " ".join((" ".join(sentence)).split()), "vid_id": video})
beam_captions.write(json.dumps(beam_annotations, indent=4, sort_keys=True))
beam_captions.close()

preds = greedy_search(beam_model, X_prev_words_begin, [X_ent_test, X_act_test, X_att_test, X_vgg_test])

greedy_captions_file = results_folder + "results/greedy_search_"+results.model_file+results.tag_type+".json"
greedy_captions = open(greedy_captions_file,"w")
greedy_annotations = []
for idx,video in enumerate(test):
        sentence = []
        for word in preds[idx]:
            sentence.append(vocabulary[word][1])
        greedy_annotations.append({"image_id": test.index(video), "caption": " ".join((" ".join(sentence)).split()), "vid_id": video})
greedy_captions.write(json.dumps(greedy_annotations, indent=4, sort_keys=True))
greedy_captions.close()

# print("python score.py -r annotations/correct_captions_ref.json -t", "results/greedy_search_"+results.model_file+results.tag_type+".json")
# print("python score.py -r annotations/correct_captions_ref.json -t", "results/beam_search_"+results.model_file+results.tag_type+".json")


# hot_captions = open("scoring_results/hot_captions_batched.sgm","w")
# print(>>hot_captions, '<tstset trglang="en" setid="y2txt" srclang="any">')
# print(>>hot_captions, '<doc sysid="langmodel" docid="y2txt" genre="vidcap" origlang="en">')
# print(>>hot_captions, '<p>')
# for idx,video in enumerate(test):
#         hot_sentence = []
#         for word in preds[idx]:
#             hot_sentence.append(vocabulary[sample(word, 0.1)][1])
#         print(>>hot_captions, '<seg id="'+str(test.index(video))+'">' + " ".join(hot_sentence) + '</seg>')
# print(>>hot_captions, '</p>')
# print(>>hot_captions, '</doc>')
# print(>>hot_captions, '</tstset>')
# hot_captions.close()
