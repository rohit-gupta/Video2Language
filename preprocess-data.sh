cd YouTube2Text/youtubeclips-dataset

# Clean Up files if they already exist
rm -f clean_descriptions.csv
rm -f matched_descriptions.csv
rm -f vocabulary_clean.txt
rm -f vidnames.txt

# Filter out clean descriptions
grep -E ",clean," video-descriptions.csv | awk -F, '{print $1"_"$2"_"$3","$8}' > clean_descriptions.csv

# Match descriptions to videos
join -t ',' <(sort youtube_mapping.txt) <(sort clean_descriptions.csv) | awk -F, '{print $2","$3}' > matched_descriptions.csv

# Create Vocabulary file
sed -e 's/<[^>]*>//g' matched_descriptions.csv | awk -F',' '{print $2}' | tr " " "\n" | sed '/^$/d' | sort | uniq -ci | sed 's/^ *//g' | sort -Vr | sed 's/ /,/g' > vocabulary_clean.txt

# Make file with list of videonames
ls *avi | sort -V > vidnames.txt

# Run Python script to generate script to extract video frames
python ../../extract_frame_gen.py > ../../extract_frames.sh

cd ../..
