cd entity_classifier
python entity_classifier_data_prep.py
cd ..

cd action_classifier
python action_classifier_data_prep.py
cd ..

cd attribute_classifier
python attribute_classifier_data_prep.py
cd ..

cd advanced_tag_models
python simple_tag_model.py -t entity -s 64 32
python simple_tag_model.py -t action -s 64 32
python simple_tag_model.py -t attribute -s 64 32
cd ..
