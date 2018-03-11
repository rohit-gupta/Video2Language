cd YouTube2Text/youtubeclips-dataset

# Clean Up files if they already exist
rm -f clean_descriptions.csv
rm -f matched_descriptions.csv
rm -f vocabulary_clean.txt
rm -f vidnames.txt

# Filter out clean descriptions if required
if [ "$1" != "CleanOnly" ]; then
    cat video-descriptions.csv | awk -F, '{print $1"_"$2"_"$3","$8}' > clean_descriptions.csv
else
    grep -E ",clean," video-descriptions.csv | awk -F, '{print $1"_"$2"_"$3","$8}' > clean_descriptions.csv
fi

# Match descriptions to videos
sed 's/ /,/g' youtube_mapping.txt > youtube_mapping.csv
join -t ',' <(sort youtube_mapping.csv) <(sort clean_descriptions.csv) | awk -F, '{print $2","$3}' > matched_descriptions.csv

# Clean descriptions
rm -f matched_descriptions_symbolfree.csv
rm -f bad_descriptions.csv 
rm -f cleaned_descriptions.csv

## Remove Symbols
sed 's/\.$//' matched_descriptions.csv | sed 's/\!$//' | sed 's/"/ /g' | tr '`' "'" | tr "[" " " | tr "]" " " | tr "/" " " |  tr "(" " " | tr ")" " " | tr "  " " " > matched_descriptions_symbolfree.csv
cat matched_descriptions_symbolfree.csv | grep "[^0-9A-Za-z,\. '&-]" > bad_descriptions.csv 
grep -v -x -f bad_descriptions.csv matched_descriptions_symbolfree.csv > symbolfree_descriptions.csv

## Remove Short Sentences
cat symbolfree_descriptions.csv | awk 'NF>=5' | sed "s/, /,/g" > cleaned_descriptions.csv
echo $(( $(wc -l video-descriptions.csv | awk '{print $1}') - $(wc -l cleaned_descriptions.csv | awk '{print $1}') )) "Captions deleted"
echo `wc -l cleaned_descriptions.csv` "Captions to be used"

# Create Vocabulary file
sed -e 's/<[^>]*>//g' cleaned_descriptions.csv | awk -F',' '{print $2}' | tr " " "\n" | sed '/^$/d' | sort | uniq -ci | sed 's/^ *//g' | sort -Vr | sed 's/ /,/g' > vocabulary.txt

# Make file with list of videonames
ls *avi | sort -V > vidnames.txt


# Run Python script to generate script to extract video frames
python ../../extract_frame_gen.py > ../../extract_frames.sh

cd ../..

mkdir logs
mkdir models

mkdir -p language_model/results
mkdir -p language_model/annotations
