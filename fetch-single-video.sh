cd deploy
youtube-dl https://www.youtube.com/watch?v=3E8nf62nihw
mkdir -p frames/vid1
ffmpeg -i Buddy_The_Rescue_Dog-3E8nf62nihw.mp4 -q:v 1 frames/vid1/%05d.jpg -hide_banner
