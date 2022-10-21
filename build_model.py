import os
from keras.callbacks import TensorBoard
from Hatewise.utilities import *

# Set dataset name
dataset_name = "hate"
# Generate model name
model_name = get_model_name(dataset_name)

# Load data
data = load_data(N_WORDS, SEQUENCE_LENGTH, TEST_SIZE, oov_token = OOV_TOKEN)

model = create_model(data["tokenizer"].word_index, units = UNITS, n_layers = N_LAYERS,
                     cell = RNN_CELL, bidirectional = IS_BIDIRECTIONAL, embedding_size = EMBEDDING_SIZE,
                     sequence_length = SEQUENCE_LENGTH, dropout = DROPOUT,
                     loss = LOSS, optimizer = OPTIMIZER, output_length = data["y_train"][0].shape[0])

model.summary()

tensorboard = TensorBoard(log_dir = os.path.join("logs", model_name))

history = model.fit(data["x_train"], data["y_train"],
                    batch_size = BATCH_SIZE,
                    epochs = EPOCHS,
                    validation_data = (data["x_test"], data["y_test"]),
                    callbacks = [tensorboard],
                    verbose = 1)

model.save(os.path.join("results", model_name) + ".h5")