import operator

# Extract visual atoms from tagged captions after the manner of "Oracle Performance for Visual Captioning" by Yao et al
fname = "Youtube2Text/youtubeclips-dataset/captions_tagged_cleaned.txt"

#Penn Treebank Project part-of-speech tags corresponding to each type of visual atom
entity_tags = ["NN", "NNP", "NNPS", "NNS", "PRP"]
action_tags = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
attribute_tags = ["JJ", "JJR", "JJS"]

entities =  {}
actions =  {}
attributes =  {}

others = []
# Read tagged captions file
with open(fname) as f:
    content = f.readlines()

tagged_captions = [x.strip() for x in content]


for caption in tagged_captions[:]:
	words = caption.split(" ")
	for tagged_word in words:
		word, tag = tagged_word.rsplit("/",1)
		if tag in entity_tags:
			# Insert into entities dict
			if word in entities:
				entities[word] += 1
			else:
				entities[word] = 1
		elif tag in action_tags:
			# actions dict
			if word in actions:
				actions[word] += 1
			else:
				actions[word] = 1
		elif tag in attribute_tags:
			# att dict
			if word in attributes:
				attributes[word] += 1
			else:
				attributes[word] = 1
		else:
			#count others
			others.append(word)

sorted_entities = reversed(sorted(entities.items(), key=operator.itemgetter(1)))
sorted_actions = reversed(sorted(actions.items(), key=operator.itemgetter(1)))
sorted_attributes = reversed(sorted(attributes.items(), key=operator.itemgetter(1)))

# 4080 Entities 2x, 3093 Entities 3x, 2266 Entities 5x
print "Generating entities ..."

entities_file = open("entities_long.txt","w")
for word,count in sorted_entities:
    print >>entities_file, word + ","  + str(count)
entities_file.close()

# 2280 actions >=2x, 1717 actions >=3x, 1227 actions >= 5x
print "Generating actions ..."
actions_file = open("actions_long.txt","w")
for word,count in sorted_actions:
    print >>actions_file, word + ","  + str(count)
actions_file.close()

# 899 Attributes >=2x, 603 Attributes >=3x
print "Generating attributes ..."
attributes_file = open("attributes_long.txt","w")
for word,count in sorted_attributes:
    print >>attributes_file, word + ","  + str(count)
attributes_file.close()