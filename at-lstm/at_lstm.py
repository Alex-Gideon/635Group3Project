import tensorflow as tf


def build_model(INPUT_SEQUENCE_LEN=80, EMBEDDING_SIZE=200, show_summary=False):
    input_tensor = tf.keras.Input(shape=(INPUT_SEQUENCE_LEN, EMBEDDING_SIZE))

    lstm1_output = tf.keras.layers.LSTM(256, return_sequences=True)(input_tensor)
    # lstm2_output = tf.keras.layers.LSTM(128, return_sequences=True)(lstm1_output)
    attention_output = tf.keras.layers.Attention()([lstm1_output, lstm1_output])

    # dense_output = tf.keras.layers.Dense(64, activation='relu')(attention_output)
    output_tensor = tf.keras.layers.Dense(1, activation='sigmoid')(attention_output)

    model = tf.keras.Model(input_tensor, output_tensor)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    if show_summary:
        model.summary()

    return model


if __name__ == '__main__':
    build_model()