import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam
import streamlit as st

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Custom Next Word Predictor", layout="wide")

st.title("🧠 Customizable Next Word Predictor Dashboard")
st.caption("Train a custom multi-layer LSTM language model and generate multi-word predictions.")

# --- Sidebar: Dynamic Model Configuration ---
st.sidebar.header("⚙️ Model & Training Hyperparameters")

# 1. File Upload (Strictly .txt)
st.sidebar.subheader("Training Data Source")
uploaded_file = st.sidebar.file_uploader("Upload Training File (.txt)", type=["txt"])

DEFAULT_TEXT = """This is a sample text for training the model. The model will learn to predict the next word based on this training data.
This is another sentence for the training. We need enough data to train a good language model.
Language modeling is an interesting task. We can generate new text after training.
Artificial intelligence is a rapidly advancing field. Machine learning is a subset of AI.
Deep learning uses neural networks. Recurrent neural networks are good for sequence data."""

if uploaded_file is not None:
    user_data = uploaded_file.read().decode("utf-8")
    st.sidebar.success(f"Loaded: {uploaded_file.name}")
else:
    user_data = DEFAULT_TEXT
    st.sidebar.info("Using default text corpus. Upload a .txt file to customize.")

# 2. Dynamic Architecture Settings
st.sidebar.subheader("Architecture Settings")
embedding_dim = st.sidebar.select_slider("Embedding Dimension", options=[16, 32, 64, 128, 256], value=64)

# Variable LSTM layers selector (1 to 5)
num_lstm_layers = st.sidebar.slider("Number of LSTM Layers", min_value=1, max_value=5, value=2)

lstm_units_list = []
for i in range(num_lstm_layers):
    default_units = 64 if i == 0 else max(16, 64 // (2 ** i))
    units = st.sidebar.slider(f"LSTM Layer {i + 1} Units", min_value=16, max_value=256, value=default_units, step=16)
    lstm_units_list.append(units)

dropout_rate = st.sidebar.slider("Dropout Rate", min_value=0.0, max_value=0.5, value=0.1, step=0.05)

# 3. Training Settings
st.sidebar.subheader("Training Settings")
learning_rate = st.sidebar.select_slider("Learning Rate", options=[0.0001, 0.001, 0.005, 0.01], value=0.01)
epochs = st.sidebar.number_input("Epochs", min_value=5, max_value=200, value=50, step=5)
batch_size = st.sidebar.select_slider("Batch Size", options=[8, 16, 32, 64], value=16)


# --- Keras Log Streaming Callback ---
class StreamlitTrainingCallback(Callback):
    def __init__(self, log_placeholder, total_epochs):
        super().__init__()
        self.log_placeholder = log_placeholder
        self.total_epochs = total_epochs
        self.log_text = ""

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        log_line = f"Epoch {epoch + 1}/{self.total_epochs} - loss: {logs.get('loss', 0):.4f} - accuracy: {logs.get('accuracy', 0):.4f}\n"
        self.log_text += log_line
        self.log_placeholder.code(self.log_text)


# --- Preprocessing Function ---
def prepare_sequences(text_corpus):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text_corpus])
    total_words = len(tokenizer.word_index) + 1

    input_sequences = []
    for line in text_corpus.split('\n'):
        cleaned_line = line.strip().lower()
        if not cleaned_line:
            continue
        token_list = tokenizer.texts_to_sequences([cleaned_line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    if not input_sequences:
        return None, None, None, 0, None

    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    X = input_sequences[:, :-1]
    y = input_sequences[:, -1]
    y = to_categorical(y, num_classes=total_words)

    return X, y, max_sequence_len, total_words, tokenizer


# --- Main Layout Tabs ---
tab1, tab2 = st.tabs(["🏗️ 1. Model Architecture & Training", "🔮 2. Multi-Word Inference"])

# --- TAB 1: MODEL BUILDING & TRAINING ---
with tab1:
    st.header("Pipeline Setup & Training")
    
    if st.button("🚀 Build & Train Model", type="primary"):
        with st.spinner("Preprocessing text data..."):
            X, y, max_seq_len, total_words, tokenizer = prepare_sequences(user_data)
            
            if X is None or len(X) == 0:
                st.error("Invalid or empty training file. Please upload a valid text file.")
                st.stop()

        # Build Dynamic Multi-Layer Model
        model = Sequential()
        model.add(Embedding(total_words, embedding_dim, input_length=max_seq_len - 1))
        
        # Add variable number of LSTM layers
        for idx, units in enumerate(lstm_units_list):
            is_last_layer = (idx == len(lstm_units_list) - 1)
            model.add(LSTM(units, return_sequences=not is_last_layer))
            if dropout_rate > 0:
                model.add(Dropout(dropout_rate))
            
        model.add(Dense(total_words, activation='softmax'))

        optimizer = Adam(learning_rate=learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # Build model weights with dummy input to calculate exact parameter shapes
        model.build(input_shape=(None, max_seq_len - 1))

        # Show Architectural Summary & Weights Breakdown
        st.subheader("📋 Architectural & Parameter Breakdown")
        
        summary_data = []
        for layer in model.layers:
            summary_data.append({
                "Layer Name": layer.name,
                "Layer Type": layer.__class__.__name__,
                "Output Shape": str(layer.output_shape),
                "Param #": f"{layer.count_params():,}"
            })
        
        st.table(pd.DataFrame(summary_data))
        
        col1, col2 = st.columns(2)
        col1.metric("Total Parameters", f"{model.count_params():,}")
        col2.metric("Trainable Parameters", f"{sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")

        # Active Training Monitor
        st.subheader("Active Training Monitor")
        log_placeholder = st.empty()
        callback = StreamlitTrainingCallback(log_placeholder, epochs)

        with st.spinner("Training model..."):
            model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=[callback], verbose=0)

        # Save model state
        st.session_state['trained_model'] = model
        st.session_state['tokenizer'] = tokenizer
        st.session_state['max_seq_len'] = max_seq_len
        st.success("🎉 Model trained successfully! Switch to the **Inference** tab to test predictions.")

# --- TAB 2: INFERENCE & MULTI-WORD PREDICTION ---
with tab2:
    st.header("Predict Next Words")
    
    if 'trained_model' not in st.session_state:
        st.info("👈 Please train a model first in the **Model Architecture & Training** tab.")
    else:
        model = st.session_state['trained_model']
        tokenizer = st.session_state['tokenizer']
        max_seq_len = st.session_state['max_seq_len']

        index_to_word = {index: word for word, index in tokenizer.word_index.items()}

        seed_text = st.text_input("Enter starting word(s):", value="this is")
        words_to_predict = st.slider("Number of words to predict:", min_value=1, max_value=20, value=10)

        if st.button("🔮 Predict Next Words"):
            if not seed_text.strip():
                st.warning("Please enter a starting word or phrase.")
            else:
                current_text = seed_text.strip()
                predicted_words = []

                progress_bar = st.progress(0)
                
                for step in range(words_to_predict):
                    token_list = tokenizer.texts_to_sequences([current_text])[0]
                    token_list = pad_sequences([token_list], maxlen=max_seq_len - 1, padding='pre')
                    
                    predicted_probs = model.predict(token_list, verbose=0)
                    predicted_index = np.argmax(predicted_probs, axis=-1)[0]
                    
                    predicted_word = index_to_word.get(predicted_index, "")
                    if not predicted_word:
                        break
                        
                    current_text += " " + predicted_word
                    predicted_words.append(predicted_word)
                    progress_bar.progress((step + 1) / words_to_predict)

                st.subheader("Generated Output:")
                st.markdown(f"**Seed Prompt:** _{seed_text}_")
                st.markdown(f"**Predicted Sequence:** **{current_text}**")
                
                st.write("---")
                st.write("### Predicted Words List:")
                st.write(" ➔ ".join(predicted_words))
