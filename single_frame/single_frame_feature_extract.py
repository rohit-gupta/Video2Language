from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import glob
import pickle
from __future__ import print_function


folder = "Youtube2Text/youtubeclips-dataset/"
single_frame = "single_frame/*"


video_frames = glob.glob(folder + single_frame)


# create feature vectors for each video
video_feature_vectors = {}

# Load the VGG model
model = VGG16(weights='imagenet', include_top=False, pooling='avg')

for video_frame in video_frames:
	img_path = video_frame
	vid_num = (img_path.split("/")[-1]).split(".")[0]
	print(vid_num)
	img = image.load_img(img_path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	features = model.predict(x)
	video_feature_vectors[vid_num] = features

pickle.dump(video_feature_vectors, open("single_frame_features.pickle", "wb"))
