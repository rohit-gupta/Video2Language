from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from __future__ import print_function

import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')

# set up file names and pathes
annFile = "annotations/correct_captions_ref.json"
resFile = "results/beam_search_medvoc_simplepredtags_batch128_lowdropout_avgfeat_threshold_minvallosspredicted.json"

# create coco object and cocoRes object
coco = COCO(annFile)
cocoRes = coco.loadRes(resFile)

# create cocoEval object by taking coco and cocoRes
cocoEval = COCOEvalCap(coco, cocoRes)

# evaluate results
cocoEval.evaluate()

# printoutput evaluation scores
for metric, score in cocoEval.eval.items():
    print('%s: %.3f'%(metric, score))
