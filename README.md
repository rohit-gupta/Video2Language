# V2L-MSVD
Generating video descriptions using deep learning in Keras

*Start with AWS Ubuntu Deep Learning AMI on a EC2 p2.xlarge instance. (or better, p2.xlarge costs $0.9/hour on-demand and ~$0.3/hour as a spot instance)*

```shell
source activate tensorflow_p27
conda install scikit-learn
conda install scikit-image
git clone https://github.com/rohit-gupta/V2L-MSVD.git
cd V2L-MSVD
```

## Using a pre-trained video captioning model

```shell
bash fetch-pretrained-model.sh
bash fetch-youtube-video.sh https://www.youtube.com/watch?v=cKWuNQAy2Sk
bash process-youtube-video.sh 
```


## Training your own video captioning model

#### Download data: should take about 2 minutes
```shell
bash fetch-data.sh
```

#### Preprocess text data: ETA ~5 minutes

If you only want to use Verified descriptions -> 

```shell
bash preprocess-data.sh CleanOnly 
```

If you want to use both verified and unverified descriptions -> 

```shell
bash preprocess-data.sh
```


#### Extract frames from the Videos: ETA ~30 minutes
```shell
bash extract_frames.sh
```

#### Extract Video Features: ETA ~15 Minutes 
```shell
bash run-feature-extractor.sh
```

#### Tag Model: ETA ~5 Minutes
```shell
bash run-simple-tag-prediction-model.sh
```
#### Train Language Model: ETA ~50 minutes (Can be killed around ~25 minutes after 5 Epochs)
```shell
bash run-language-model.sh
```

#### Score Language Model: ETA ~5 minutes
```shell
bash score-language-model.sh
```

## Known Issues

- If at any stage you get an error that contains 

```shell
/lib/libstdc++.so.6: version `CXXABI_1.3.x' not found
```

You can fix it with:

```shell
cd ~/anaconda3/envs/tensorflow_p27/lib && mv libstdc++.a stdcpp_bkp && mv libstdc++.so stdcpp_bkp && mv libstdc++.so.6 stdcpp_bkp && mv libstdc++.so.6.0.19 stdcpp_bkp/  && mv libstdc++.so.6.0.19-gdb.py stdcpp_bkp/  && mv libstdc++.so.6.0.21 stdcpp_bkp/  && mv libstdc++.so.6.0.24 stdcpp_bkp/ && cd -
```

- Tensorflow 1.3 has a memory leak bug that might affect this code

You can fix it by upgrading Tensorflow. 

Reference for this problem: https://github.com/rohit-gupta/Video2Language/issues/3