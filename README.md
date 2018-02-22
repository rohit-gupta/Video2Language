# V2L-MSVD
Generating video descriptions using deep learning in Keras

## Start with AWS Ubuntu Deep Learning AMI on a EC2 p2.xlarge instance. (or better, p2.xlarge costs $0.9/hour on-demand and ~$0.3/hour as a spot instance)

```shell
source activate tensorflow_p27
conda install scikit-learn
conda install scikit-image
git clone https://github.com/rohit-gupta/V2L-MSVD.git
cd V2L-MSVD
```

### Download data: should take about 2 minutes
```shell
bash fetch-data.sh
```

### Preprocess text data: ETA ~5 minutes
```shell
bash preprocess-data.sh
```

### Extract frames from the Videos: ETA ~30 minutes
```shell
bash extract_frames.sh
```

### Extract Video Features: ETA ~50 Minutes 
```shell
bash run-feature-extractor.sh
```

### Tag Model: ETA ~5 Minutes
```shell
bash run-simple-tag-prediction-model.sh
```
### Train Language Model: ETA ~20 minutes (Can be killed around ~10 minutes after 5 Epochs)
```shell
bash run-language-model.sh
```

### Score Language Model: ETA ~5 minutes
```shell
bash score-language-model.sh
```

