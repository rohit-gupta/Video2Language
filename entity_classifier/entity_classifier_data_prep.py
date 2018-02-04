import numpy as np
import pickle
import sys

folder = "../Youtube2Text/youtubeclips-dataset/"
entities_file = "../tag_generator/entities_long.txt"
captions_file = "matched_all_descriptions.csv"

video_entities = {}

# read in a list of entities
with open(entities_file) as f:
    entities_temp = f.readlines()

entities_all = [x.strip().split(',') for x in entities_temp]
MIN_COUNT=50
entities = []
for word,count in entities_all:
	if int(count) >= MIN_COUNT:
		entities.append(word) 

print "Using " + str(len(entities)) + " entities out of total " + str(len(entities_all)) + " entities"

# read in cleaned up captions
with open(folder+captions_file) as f:
    captions_temp = f.readlines()

captions = [x.strip() for x in captions_temp]

for caption in captions:
	video, caption_text = caption.split(",")
	caption_words = caption_text.split(' ')
	for word in caption_words:
		if word in entities:
			if video in video_entities:
				video_entities[video].append(word)
			else:
				video_entities[video] = [word]


# remove duplicates
for video in video_entities.keys():
	video_entities[video] = list(set(video_entities[video]))

# num_entities = []

# for video in video_entities.keys():
# 	num_entities.append(len(video_entities[video]))

# create entity vectors for each video
video_entity_vectors = {}

for video in video_entities.keys():
	entity_vector = np.zeros(len(entities),dtype=np.float32)
	for entity in video_entities[video]:
		entity_vector[entities.index(entity)] = 1.0
	video_entity_vectors[video] = entity_vector

# save entity vectors to disk
pickle.dump(video_entity_vectors, open("entity_vectors_long.pickle", "wb"))
