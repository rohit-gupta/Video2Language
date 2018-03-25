cd deploy

if [ "$2" != "NoCleanup" ]; then
	rm vid* *pickle && rm -rf frames/
fi

youtube-dl $1 -o 'vid1.%(ext)s'
mkdir -p frames/vid1
ffmpeg -i vid1.m* -q:v 1 frames/vid1/%05d.jpg -hide_banner
cd ..
