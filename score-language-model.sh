cd language_model
git clone https://github.com/tylin/coco-caption.git
python generate.py -m medvoc_simplepredtags_batch128_lowdropout_avgfeat_threshold_minvalloss -t predicted
mv annotations/* coco-caption/annotations/
mv results/* coco-caption/results/
cp score.py coco-caption 
cd coco-caption
python score.py
cd ../..
