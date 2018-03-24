cd language_model
git clone https://github.com/tylin/coco-caption.git
mkdir -p results
python generate.py -p predicted -t 0.01 -s 512 -m medvoc_simplepredtags_batch128_lowdropout_avgfeat_threshold_maxvalacc
mv annotations/* coco-caption/annotations/
mv results/* coco-caption/results/
cp score.py coco-caption 
cd coco-caption
python score.py
cd ../..
