import nltk
from __future__ import print_function

fname = "correct_captions.txt"
with open(fname) as f:
    content = f.readlines()

correct = [x.strip() for x in content]


fname = "greedy_captions.txt"
with open(fname) as f:
    content = f.readlines()

greedy = [x.strip() for x in content]

average = 0.0
max_score = 0.0
max_idx = 0
for idx in range(len(correct)):
	hypothesis = greedy[idx].split(" ")
	reference = correct[idx].split(" ")
	BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis)
	if BLEUscore > max_score:
		max_score = BLEUscore
		max_idx = idx
	average += BLEUscore

print("BLEU 4", average/len(correct))

print(greedy[max_idx])
print(correct[max_idx])
