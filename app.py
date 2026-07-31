import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
import streamlit as st
import io
import contextlib


st.set_page_config(page_title="Next Word Predictor", layout="wide")

# --- Keras Callback for Streamlit Log Streaming ---
class StreamlitTrainingCallback(Callback):
    def __init__(self, log_placeholder):
        super().__init__()
        self.log_placeholder = log_placeholder
        self.log_text = ""

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Format the log line just like you'd see in the console
        log_line = f"Epoch {epoch+1}/50 - loss: {logs.get('loss'):.4f} - accuracy: {logs.get('accuracy'):.4f}\n"
        self.log_text += log_line
        # Dynamically update the placeholder container in the UI
        self.log_placeholder.code(self.log_text)

# --- Data Initialization ---
data = ("This is a sample text for training the model. The model will learn to predict the next word based on this training data."
        "This is another sentence for the training. We need enough data to train a good language model."
        "Language modeling is an interesting task. We can generate new text after training."
        "Artificial intelligence is a rapidly advancing field. Machine learning is a subset of AI."
        "Deep learning uses neural networks. Recurrent neural networks are good for sequence data.")

# We cache the tokenizer so it doesn't re-run every time Streamlit rerenders
@st.cache_resource
def get_tokenizer(text_data):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text_data])
    return tokenizer

tokenizer = get_tokenizer(data)

# --- Helper Functions ---
def pre_build(doc):
    input_sequence = []
    for sentence in doc.split('\n'):
        if not isinstance(sentence, str):
            continue
        cleaned_sentence = sentence.strip().lower()
        if not cleaned_sentence:
            continue
        tokenize_sentence = tokenizer.texts_to_sequences([cleaned_sentence])[0]
        for i in range(1, len(tokenize_sentence)):
            input_sequence.append(tokenize_sentence[:i+1])

    if not input_sequence:
        return np.array([]), np.array([]), 0

    max_len_local = max([len(x) for x in input_sequence])
    padded_input_sequence = np.array(pad_sequences(input_sequence, maxlen=max_len_local, padding='pre'))
    X = padded_input_sequence[:, :-1]
    y = padded_input_sequence[:, -1]
    y = to_categorical(y, num_classes=len(tokenizer.word_index) + 1)
    return X, y, max_len_local

def generate_next_words(model, max_sequence_len, input_text, num_words_to_generate=10):
    if model is None or max_sequence_len == 0:
        return "Model not trained yet. Go to the training tab first!"

    text = input_text.lower().strip()
    if not text:
        return "Please enter some text to start generation."

    generated_text = text
    for _ in range(num_words_to_generate):
        token_text = tokenizer.texts_to_sequences([text])[0]
        prediction_input_length = max_sequence_len - 1

        if not prediction_input_length or prediction_input_length < 1:
            return "Error: Model's expected input length is invalid."

        padded_token_text = pad_sequences([token_text], maxlen=prediction_input_length, padding='pre')
        if padded_token_text.shape[1] == 0:
             return "Error: Input text too short."

        prediction_probabilities = model.predict(padded_token_text, verbose=0)
        if prediction_probabilities.size == 0:
            return "Error: Model prediction failed."
        
        pos = np.argmax(prediction_probabilities[0])
        found_word = False
        for word, index in tokenizer.word_index.items():
            if index == pos:
                generated_text += " " + word
                text += " " + word
                found_word = True
                break
        if not found_word:
            generated_text += " <UNK>"
            text += " <UNK>"

    return generated_text

# --- Persistent Session State Initialization ---
if 'model' not in st.session_state:
    st.session_state.model = None
if 'max_sequence_len' not in st.session_state:
    st.session_state.max_sequence_len = 0

# --- Streamlit UI Architecture ---
st.title("Next Word Predictor Dashboard")
st.write("Train models dynamically and perform inference using your text samples.")

tab1, tab2 = st.tabs(["🏗️ 1. Model Summary & Training Logs", "🔮 2. Inference & Prediction"])

with tab1:
    st.header("Model Pipeline Workspace")
    
    if st.button("🚀 Start Training Pipeline", type="primary"):
        st.write("### 📜 Pipeline Output Log")
        status_log = st.empty()
        
        # 1. Prebuild Data Phase
        status_log.info("Preparing data collections...")
        X_train, y_train, max_len_from_prebuild = pre_build(data)
        
        if X_train.size > 0 and y_train.size > 0 and max_len_from_prebuild >= 2:
            st.session_state.max_sequence_len = max_len_from_prebuild
            
            # 2. Build Model Configuration
            status_log.info("Compiling model graph...")
            model = Sequential([
                Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100),
                LSTM(200, return_sequences=True),
                LSTM(200),
                Dense(len(tokenizer.word_index) + 1, activation='softmax')
            ])
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            # Render model summary natively in Streamlit
            st.write("#### Architectural Summary")
            summary_buffer = io.StringIO()
            model.summary(print_fn=lambda x: summary_buffer.write(x + '\n'))
            st.code(summary_buffer.getvalue())
            
            # 3. Model Fit Loop Execution
            st.write("#### Active Training Monitor")
            training_log_ui = st.empty()
            
            streamlit_callback = StreamlitTrainingCallback(training_log_ui)
            
            status_log.warning("Training steps running...")
            model.fit(X_train, y_train, epochs=50, verbose=0, callbacks=[streamlit_callback])
            
            # Update state with trained model
            st.session_state.model = model
            status_log.success("🎉 Pipeline optimization completed successfully!")
        else:
            status_log.error("Insufficient validation sequence data detected.")

with tab2:
    st.header("Inference Playground")
    
    # Check if model exists before handling operations cleanly
    if st.session_state.model is None:
        st.warning("⚠️ Warning: No active model detected in memory. Please complete the process inside Tab 1 first.")
    
    # Input options
    uploaded_file = st.file_uploader("Option A: Upload a text file", type=["txt"])
    input_text = st.text_area("Option B: Enter seed prompt text manually", height=100)
    
    if st.button("✨ Predict Next 10 Words"):
        final_input = ""
        
        if uploaded_file is not None:
            try:
                final_input = uploaded_file.read().decode("utf-8")
            except Exception as e:
                st.error(f"Failed parsing documentation dataset: {e}")
        elif input_text.strip():
            final_input = input_text
            
        if final_input:
            with st.spinner("Executing sequence probabilities..."):
                prediction_result = generate_next_words(
                    st.session_state.model, 
                    st.session_state.max_sequence_len, 
                    final_input
                )
            st.write("### Continuation Response:")
            st.success(prediction_result)
        else:
            st.error("Please supply a seed text prompt or drop a plain text document to proceed.")