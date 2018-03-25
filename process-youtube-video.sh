cd deploy
python extract_video_features.py -v frames/vid1 -f $(echo "scale=4;"`ffprobe -v 0 -of csv=p=0 -select_streams 0 -show_entries stream=r_frame_rate vid1.m*` | bc)
python predict_caption.py
cd ..
