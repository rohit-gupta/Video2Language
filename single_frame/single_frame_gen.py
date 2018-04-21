from __future__ import print_function

for num in range(1970):
	print('ffmpeg -i vid%d.avi -vf "select=gte(n\,10)" -vframes 1 single_frame/vid%d.png' % (num+1, num+1))
