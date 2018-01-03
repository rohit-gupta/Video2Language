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

# Add meta characters to vocabulary
vocabulary = full_vocabulary[:NUM_WORDS]
shuffle(vocabulary) # shuffles in-place
vocabulary.append((0,"<BOS>"))
vocabulary.append((0,"<EOS>"))
vocabulary = [(0,"<unk>")] + vocabulary
pickle.dump(vocabulary, open("vocabulary.p", "wb"))

# Turn vocabulary into list of words
vocabulary = [x[1] for x in vocabulary]


# Load Captions
#fname = folder + "descriptions_normalized.csv"
fname = folder + "matched_unverified_descriptions.csv"
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
print "MEDIAN_CAPTION_LEN: ", TRUNCATED_CAPTION_LEN
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
		encoded_captions.append((video, one_hot_encode(['<BOS>'] + caption.split(" ") + ['<EOS>'],vocabulary)))

pickle.dump(encoded_captions, open("encoded_captions_unverified_len_4_10.p", "wb"))

# for idx in range(10):
# 	coded_string = []
# 	for encoded_word in encoded_captions[idx][1]:
# 		coded_string.append(vocabulary[np.argmax(encoded_word)])
# 	print " ".join(coded_string)