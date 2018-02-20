import numpy as np
import pickle
import sys

folder = "../YouTube2Text/youtubeclips-dataset/"
actions_file = "../tag_generator/actions_long.txt"
captions_file = "cleaned_descriptions.csv"

video_actions = {}

# read in a list of actions
with open(actions_file) as f:
    actions_temp = f.readlines()

actions_all = [x.strip().split(',') for x in actions_temp]
MIN_COUNT=30
actions = []
for word,count in actions_all:
	if int(count) >= MIN_COUNT:
		actions.append(word) 

print "Using " + str(len(actions)) + " actions out of total " + str(len(actions_all)) + " actions"

# read in cleaned up captions
with open(folder+captions_file) as f:
    captions_temp = f.readlines()

captions = [x.strip() for x in captions_temp]

for caption in captions:
	video, caption_text = caption.split(",")
	caption_words = caption_text.split(' ')
	for word in caption_words:
		if word in actions:
			if video in video_actions:
				video_actions[video].append(word)
			else:
				video_actions[video] = [word]


# remove duplicates
for video in video_actions.keys():
	video_actions[video] = list(set(video_actions[video]))

# num_actions = []

# for video in video_actions.keys():
# 	num_actions.append(len(video_actions[video]))

# create action vectors for each video
video_action_vectors = {}

for video in video_actions.keys():
	action_vector = np.zeros(len(actions),dtype=np.float32)
	for action in video_actions[video]:
		action_vector[actions.index(action)] = 1.0
	video_action_vectors[video] = action_vector

# save action vectors to disk
pickle.dump(video_action_vectors, open("action_vectors_long.pickle", "wb"))
