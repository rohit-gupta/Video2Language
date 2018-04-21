import numpy as np
import pickle
import sys
from __future__ import print_function

folder = "../YouTube2Text/youtubeclips-dataset/"
attributes_file = "../tag_generator/attribute_long.txt"
captions_file = "cleaned_descriptions.csv"

video_attributes = {}

# read in a list of attributes
with open(attributes_file) as f:
    attributes_temp = f.readlines()

attributes_all = [x.strip().split(',') for x in attributes_temp]
MIN_COUNT=25
attributes = []
for word,count in attributes_all:
	if int(count) >= MIN_COUNT:
		attributes.append(word)

print("Using " + str(len(attributes)) + " attributes out of total " + str(len(attributes_all)) + " attributes")

# read in cleaned up captions
with open(folder+captions_file) as f:
    captions_temp = f.readlines()

captions = [x.strip() for x in captions_temp]

for caption in captions:
	video, caption_text = caption.split(",")
	caption_words = caption_text.split(' ')
	for word in caption_words:
		if word in attributes:
			if video in video_attributes:
				video_attributes[video].append(word)
			else:
				video_attributes[video] = [word]


# remove duplicates
for video in video_attributes.keys():
	video_attributes[video] = list(set(video_attributes[video]))

# num_attributes = []

# for video in video_attributes.keys():
# 	num_attributes.append(len(video_attributes[video]))

# create attribute vectors for each video
video_attribute_vectors = {}

for video in video_attributes.keys():
	attribute_vector = np.zeros(len(attributes),dtype=np.float32)
	for attribute in video_attributes[video]:
		attribute_vector[attributes.index(attribute)] = 1.0
	video_attribute_vectors[video] = attribute_vector

# save attribute vectors to disk
pickle.dump(video_attribute_vectors, open("attribute_vectors_long.pickle", "wb"))
