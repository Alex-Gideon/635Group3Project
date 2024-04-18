import tensorflow as tf


def build_model(INPUT_SEQUENCE_LEN=80, EMBEDDING_SIZE=200, show_summary=False):
    input_tensor = tf.keras.Input(shape=(INPUT_SEQUENCE_LEN, EMBEDDING_SIZE))

    lstm_output = tf.keras.layers.LSTM(64, return_sequences=True)(input_tensor)
    attention_output = tf.keras.layers.Attention()([lstm_output, lstm_output])

    dense_output = tf.keras.layers.Dense(64, activation='relu')(attention_output)
    output_tensor = tf.keras.layers.Dense(1, activation='sigmoid')(dense_output)

    model = tf.keras.Model(input_tensor, output_tensor)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    if show_summary:
        model.summary()

    return model
