import tensorflow as tf


def build_model(INPUT_SEQUENCE_LEN=80, EMBEDDING_SIZE=200, show_summary=False):
    """constructs the AT-LSTM model

    Args:
        INPUT_SEQUENCE_LEN (int, optional): maximum input sentence length. Defaults to 80.
        EMBEDDING_SIZE (int, optional): embedding size of each word. Defaults to 200.
        show_summary (bool, optional): flag to print summary of the model's architecture. Defaults to False.

    Returns:
        tf model: at-lstm model
    """
    input_tensor = tf.keras.Input(shape=(INPUT_SEQUENCE_LEN, EMBEDDING_SIZE))

    lstm1_output = tf.keras.layers.LSTM(256, return_sequences=True)(input_tensor)
    lstm2_output = tf.keras.layers.LSTM(256, return_sequences=True)(lstm1_output)
    attention_output = tf.keras.layers.Attention()([lstm1_output, lstm2_output])

    # dense_output = tf.keras.layers.Dense(64, activation='relu')(attention_output)
    output_tensor = tf.keras.layers.Dense(1, activation='sigmoid')(attention_output)

    model = tf.keras.Model(input_tensor, output_tensor)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    if show_summary:
        model.summary()

    return model


if __name__ == '__main__':
    build_model()