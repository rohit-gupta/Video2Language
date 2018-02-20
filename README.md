# V2L-MSVD
Generating video descriptions using deep learning

```shell
source activate tensorflow_p27
conda install scikit-learn
git clone https://github.com/rohit-gupta/V2L-MSVD.git
cd V2L-MSVD
```

### Download data: should take about 2 minutes
```shell
bash fetch-data.sh
```

### Preprocess text data: ETA: ~5 minutes
```shell
bash preprocess-data.sh
```

### Extract frames from the Videos: ETA: ~30 minutes
```shell
bash extract_frames.sh
```

### ETA: ~50 Minutes 
```shell
bash run-feature-extractor.sh
```

### Tag Model
```shell
bash run-simple-tag-prediction-model.sh
```
### Language Model
```shell
bash run-language-model.sh
```
