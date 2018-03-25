cd deploy

if [ "$2" != "NoCleanup" ]; then
	rm vid* *pickle && rm -rf frames/
fi

mkdir -p frames/vid1
ffmpeg -i $1 -q:v 1 frames/vid1/%05d.jpg -hide_banner
cd ..
