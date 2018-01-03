#from __future__ import print_function
import imageio
import glob
import sys

videos = glob.glob("Youtube2Text/youtubeclips-dataset/*.avi")

print("Filename;fps;nframes;size;source_size")
for video in videos:
	reader = imageio.get_reader(video)
	metadata = reader.get_meta_data()
	if metadata['fps'] * 3 > metadata['nframes']:
		print video.split("/")[-1]
