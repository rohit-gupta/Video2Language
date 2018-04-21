import numpy as np
import pickle
import argparse
from __future__ import print_function

parser = argparse.ArgumentParser(description='Generate Bleu Score on YouTube2Text.')
parser.add_argument(dest='testfile', metavar='XYZ.tst', type=argparse.FileType('r'), help='Generated Captions file to test')
args = parser.parse_args()

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

vid_mapping = []
for idx,video in enumerate(test):
	vid_mapping.append((idx,video))



with open("scoring_results/correct_captions_ref.sgm","r") as f:
	content = f.readlines()

correct_captions = [x.strip() for x in content]
correct_captions = correct_captions[3:-3] # chop off markup metadata first and last 3 lines

references = [0]*len(test)
ref_count = [0]*len(test)
for caption in correct_captions:
	vid_id = int(caption.split('"')[1])
	text = caption.split('<')[1].split('>')[1].strip()
	if ref_count[vid_id] == 0:
		references[vid_id] = (vid_mapping[vid_id],[text])
		ref_count[vid_id] += 1
	else:
		references[vid_id][1].append(text)


#with open("scoring_results/" + test_file + ".sgm","r") as f:
content = args.testfile.readlines()

greedy_captions = [x.strip() for x in content]
greedy_captions = greedy_captions[3:-3] # chop off markup metadata first and last 3 lines

hypotheses = [0]*len(test)
for caption in greedy_captions:
	vid_id = int(caption.split('"')[1])
	text = caption.split('<')[1].split('>')[1].strip()
	hypotheses[vid_id] = (vid_mapping[vid_id],text)

from nltk.translate.bleu_score import corpus_bleu,sentence_bleu

list_of_references_bleu = []
hypotheses_bleu = []

#print("VideoID,Bleu4Score")
# iterate over each test video
for idx in range(len(hypotheses)):
	for refno,reference in enumerate(references):
		if references[refno][0][1] == hypotheses[idx][0][1]: # Match video IDs
			#printhypotheses[idx][0][1], sentence_bleu(references[refno][1], hypotheses[idx][1])
			list_of_references_bleu.append(references[refno][1])
			hypotheses_bleu.append(hypotheses[idx][1])

#print("Corpus,", corpus_bleu(list_of_references_bleu, hypotheses_bleu))

print("B@1,B@2,B@3,B@4", corpus_bleu(list_of_references_bleu, hypotheses_bleu,weights = [1.0]), corpus_bleu(list_of_references_bleu, hypotheses_bleu,weights = (0.5, 0.5)),corpus_bleu(list_of_references_bleu, hypotheses_bleu,weights = (0.33333333, 0.33333333, 0.33333333)),corpus_bleu(list_of_references_bleu, hypotheses_bleu))
