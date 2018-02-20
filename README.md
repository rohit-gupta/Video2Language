# V2L-MSVD
Generating video descriptions using deep learning

source activate tensorflow_p27
git clone https://github.com/rohit-gupta/V2L-MSVD.git
cd V2L-MSVD

# Should take about 2 minutes
bash fetch-data.sh

# ETA: ~5 minutes
bash preprocess-data.sh

# ETA: ~30 minutes
bash extract_frames.sh

# ETA: ~1hr 
bash run-feature-extractor.sh

# Tag Model

# Language Model
