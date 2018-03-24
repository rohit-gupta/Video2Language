# import keras
import keras # For keras.layers.concatenate
from keras.layers import Input, Dense, RepeatVector, Embedding
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.optimizers import RMSprop


def get_language_model(NUM_PREV_WORDS, VOCABULARY_SIZE, EMBEDDING_DIM, NUM_FEATURES, NUM_ENTITIES, NUM_ACTIONS, NUM_ATTRIBUTES, LSTM_SIZE, LSTM_DROPOUT, LSTM_RECURRENT_DROPOUT):
    # Language Model
    previous_words_input = Input(shape=(NUM_PREV_WORDS,), dtype='float32')
    previous_words_embedded = Embedding(output_dim=EMBEDDING_DIM, input_dim=VOCABULARY_SIZE, mask_zero=True, dtype='float32')(previous_words_input)
    lstm1_output = LSTM(LSTM_SIZE, return_sequences=True, implementation=2, dropout=LSTM_DROPOUT, recurrent_dropout=LSTM_RECURRENT_DROPOUT)(previous_words_embedded)
    words_embedding_space = TimeDistributed(Dense(EMBEDDING_DIM))(lstm1_output)
    
    # Features Model
    frame_features_input = Input(shape=(NUM_FEATURES,),   dtype='float32')
    features_embedding_space = Dense(EMBEDDING_DIM, activation='relu', dtype='float32')(frame_features_input)
    features_embedding_space_repeated = RepeatVector(NUM_PREV_WORDS)(features_embedding_space)
    
    # Tags Model
    entities_input     = Input(shape=(NUM_ENTITIES,),   dtype='float32')
    actions_input      = Input(shape=(NUM_ACTIONS,),    dtype='float32')
    attributes_input   = Input(shape=(NUM_ATTRIBUTES,), dtype='float32')
    tags_merged = keras.layers.concatenate([entities_input, actions_input, attributes_input])
    tags_embedding_space = RepeatVector(NUM_PREV_WORDS)(tags_merged)
    
    # Word Generation Model
    lang_model_input = keras.layers.concatenate([words_embedding_space, features_embedding_space_repeated, tags_embedding_space])
    lstm_output = LSTM(512, return_sequences=False, implementation=2, dropout=0.5, recurrent_dropout=0.5)(lang_model_input) # fast_gpu implementation
    generated_word = Dense(VOCABULARY_SIZE, activation='softmax')(lstm_output)
    
    
    beam_model = Model(inputs=[entities_input, actions_input, attributes_input, frame_features_input, previous_words_input], outputs=generated_word)
    beam_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    beam_model.summary()
    return beam_model