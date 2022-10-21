import numpy as np
from tqdm import tqdm
from keras.layers import Dense, Dropout, LSTM, Embedding, Bidirectional
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Maximum number of words per entry
SEQUENCE_LENGTH = 200
# Embedding vectors
EMBEDDING_SIZE = 200
# Total number of words to sample
N_WORDS = 100000
# Out-of-vocabulary token
OOV_TOKEN = None
# Test size ratio (testing vs. training)
TEST_SIZE = 0.5
# Number of N-layers
N_LAYERS = 1
# RNN cell type
RNN_CELL = LSTM
# Whether cell is bidirectional
IS_BIDIRECTIONAL = False
# Number of nodes per layer
UNITS = 128
# Data dropout rate
DROPOUT = 0.4
# Training parameters
LOSS = "categorical_crossentropy"
OPTIMIZER = "adam"
BATCH_SIZE = 64
EPOCHS = 20


def get_model_name(dataset_name):
    # Assign model name
    model_name = f"{dataset_name}-{RNN_CELL.__name__}"
    if IS_BIDIRECTIONAL:
        model_name = "bid-" + model_name
    if OOV_TOKEN:
        model_name += "-oov"
    return model_name


def get_embedding_vectors(word_index, embedding_size=100):
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_size))
    with open(f"data/glove.6B.{embedding_size}d.txt", encoding = "utf8", errors = "ignore") as f:
        for line in tqdm(f):
            values = line.split()
            # Retrieve first vocab
            word = values[0]
            if word in word_index:
                idx = word_index[word]
                # Retrieve remaining vocab
                embedding_matrix[idx] = np.array(values[1:], dtype = "float32")
    return embedding_matrix


def create_model(word_index, units = 128, n_layers = 1, cell=LSTM, bidirectional=False,
                 embedding_size = 100, sequence_length = 100, dropout = 0.3,
                 loss = "categorical_crossentropy", optimizer = "adam",
                 output_length = 2):

    embedding_matrix = get_embedding_vectors(word_index, embedding_size)
    model = Sequential()
    # Created new embedded layer
    model.add(Embedding(len(word_index) + 1,
              embedding_size,
              weights = [embedding_matrix],
              trainable = False,
              input_length = sequence_length))

    for i in range(n_layers):
        if i == n_layers - 1:
            # Add final layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences = False)))
            else:
                model.add(cell(units, return_sequences = False))
        else:
            # Create first or hidden layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences = True)))
            else:
                model.add(cell(units, return_sequences = True))
        model.add(Dropout(dropout))

    model.add(Dense(output_length, activation = "softmax"))
    model.compile(optimizer = optimizer, loss = loss, metrics = ["accuracy"])
    return model


def load_data(num_words, sequence_length, test_size = 0.50, oov_token = None):
    # Read data from dataset
    dataterm = []
    with open("data/dataset.txt", encoding = 'utf-8') as f:
        for review in f:
            review = review.strip()
            dataterm.append(review)

    labels = []
    with open("data/labels.txt") as f:
        for label in f:
            label = label.strip()
            labels.append(label)

    # Text processing (tokenize words, remove stopwords)
    tokenizer = Tokenizer(num_words = num_words, oov_token = oov_token)
    tokenizer.fit_on_texts(dataterm)
    x = tokenizer.texts_to_sequences(dataterm)
    
    x, y = np.array(x), np.array(labels)

    x = pad_sequences(x, maxlen = sequence_length)
    y = to_categorical(y)

    # Divide data into training and testing subsets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size, random_state = 1)

    data = {"x_train": x_train, "x_test": x_test, "y_train": y_train, "y_test": y_test, "tokenizer": tokenizer,
            "int2label": {0: "normal speech", 1: "hateful speech"},
            "label2int": {"normal speech": 0, "hateful speech": 1}}

    return data


