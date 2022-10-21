import os
from Hatewise.utilities import *

# Dataset name and retrieve
dataset_name = "dataset"

model_name = get_model_name(dataset_name)

data = load_data(N_WORDS, SEQUENCE_LENGTH, TEST_SIZE, oov_token = OOV_TOKEN)

model = create_model(data["tokenizer"].word_index, units = UNITS, n_layers = N_LAYERS,
                     cell = RNN_CELL, bidirectional = IS_BIDIRECTIONAL, embedding_size = EMBEDDING_SIZE,
                     sequence_length = SEQUENCE_LENGTH, dropout = DROPOUT,
                     loss = LOSS, optimizer = OPTIMIZER, output_length = data["y_train"][0].shape[0])

model.load_weights(os.path.join("results", f"{model_name}.h5"))


def get_predictions(text_data):
    sequence = data["tokenizer"].texts_to_sequences([text_data])
    sequence = pad_sequences(sequence, maxlen = SEQUENCE_LENGTH)

    # Process prediction model output
    prediction = model.predict(sequence)[0]
    return data["int2label"][np.argmax(prediction)], prediction


while True:
    text = input("Enter your text: ")
    prediction = get_predictions(text)
    classification = prediction[0]
    if classification == "hateful speech":
        conf = round(prediction[1][1] * 100, 2)
    else: conf = round(prediction[1][0] * 100, 2)
    print(f"This has been rated as {classification} with {conf}% confidence.")

