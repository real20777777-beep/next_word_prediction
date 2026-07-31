
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd

# Set page config
st.set_page_config(
    page_title="Custom Next Word Predictor",
    page_icon="🧠",
    layout="wide"
)

# --- CUSTOM CSS STYLING ---
st.markdown("""
    <style>
    /* Gradient Main Header */
    .main-title {
        background: linear-gradient(90deg, #FF4B4B, #FF8C00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    /* Styled Prediction Container */
    .prediction-box {
        background: linear-gradient(135deg, #1e1e2f, #2a2a40);
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #FF4B4B;
        margin-top: 15px;
        margin-bottom: 20px;
    }
    
    .prompt-label {
        color: #888888;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }
    
    .prompt-text {
        font-size: 1.1rem;
        color: #e0e0e0;
        font-style: italic;
    }
    
    .result-text {
        font-size: 1.3rem;
        font-weight: 700;
        color: #00E676;
    }
    
    /* Custom Styling for Buttons */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #FF4B4B 0%, #FF7B54 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.25rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(255, 75, 75, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

# Title & Subtitle
st.markdown('<h1 class="main-title">🧠 Customizable Next Word Predictor Dashboard</h1>', unsafe_allow_html=True)
st.write("Train a custom multi-layer LSTM language model and generate multi-word predictions.")

# --- SIDEBAR / HYPERPARAMETERS ---
st.sidebar.header("⚙️ Model & Training Hyperparameters")

st.sidebar.subheader("Training Data Source")
data_source = st.sidebar.radio("Choose Data Input Method:", ["Paste Text Area", "Upload .txt File"])

corpus_text = ""
if data_source == "Paste Text Area":
    corpus_text = st.sidebar.text_area(
        "Training Data Corpus",
        value="This is a sample text for training the model. The model will learn to predict the next word based on this training data. "
              "This is another sentence for the training. We need enough data to train a good language model. "
              "Language modeling is an interesting task. We can generate new text after training. "
              "Artificial intelligence is a rapidly advancing field. Machine learning is a subset of AI. "
              "Deep learning uses neural networks. Recurrent neural networks are good for sequence data.",
        height=180
    )
else:
    uploaded_file = st.sidebar.file_uploader("Upload a .txt File", type=["txt"])
    if uploaded_file is not None:
        corpus_text = uploaded_file.read().decode("utf-8")

st.sidebar.subheader("Architecture Settings")
embedding_dim = st.sidebar.slider("Embedding Dimension", min_value=16, max_value=256, value=64, step=16)
num_lstm_layers = st.sidebar.slider("Number of LSTM Layers", min_value=1, max_value=5, value=2)

lstm_units_list = []
for i in range(num_lstm_layers):
    default_val = 64 if i == 0 else 32
    units = st.sidebar.slider(f"LSTM Layer {i+1} Units", min_value=16, max_value=256, value=default_val, step=16)
    lstm_units_list.append(units)

dropout_rate = st.sidebar.slider("Dropout Rate", min_value=0.0, max_value=0.5, value=0.1, step=0.05)

st.sidebar.subheader("Training Settings")
learning_rate = st.sidebar.select_slider("Learning Rate", options=[0.0001, 0.001, 0.005, 0.01, 0.05, 0.1], value=0.01)
epochs = st.sidebar.number_input("Epochs", min_value=1, max_value=500, value=50, step=5)
batch_size = st.sidebar.slider("Batch Size", min_value=8, max_value=64, value=16, step=8)


# Custom Streamlit Keras Training Callback
class StreamlitTrainingCallback(tf.keras.callbacks.Callback):
    def __init__(self, placeholder, total_epochs):
        super().__init__()
        self.placeholder = placeholder
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get('loss', 0.0)
        acc = logs.get('accuracy', 0.0)
        progress = (epoch + 1) / self.total_epochs
        self.placeholder.markdown(
            f"**Epoch {epoch + 1}/{self.total_epochs}** — "
            f"Loss: `{loss:.4f}` | Accuracy: `{acc * 100:.2f}%`"
        )


# --- MAIN TABS ---
tab1, tab2 = st.tabs(["🏗️ 1. Model Architecture & Training", "🔮 2. Multi-Word Inference"])

# --- TAB 1: TRAINING ---
with tab1:
    st.header("Pipeline Setup & Training")
    
    if st.button("🚀 Build & Train Model"):
        if not corpus_text.strip():
            st.error("Invalid or empty training content. Please enter text or upload a valid file.")
            st.stop()
            
        lines = [line.strip() for line in corpus_text.split('\n') if line.strip()]
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(lines)
        total_words = len(tokenizer.word_index) + 1

        input_sequences = []
        for line in lines:
            token_list = tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                n_gram_seq = token_list[:i+1]
                input_sequences.append(n_gram_seq)

        if not input_sequences:
            st.error("Text content is too short to generate training sequences.")
            st.stop()

        max_seq_len = max([len(x) for x in input_sequences])
        input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre'))

        X, y = input_sequences[:, :-1], input_sequences[:, -1]
        y = tf.keras.utils.to_categorical(y, num_classes=total_words)

        # Build Dynamic Model
        model = Sequential()
        model.add(Embedding(total_words, embedding_dim, input_length=max_seq_len - 1))

        for idx, units in enumerate(lstm_units_list):
            is_last_layer = (idx == len(lstm_units_list) - 1)
            model.add(LSTM(units, return_sequences=not is_last_layer))
            if dropout_rate > 0:
                model.add(Dropout(dropout_rate))

        model.add(Dense(total_words, activation='softmax'))

        optimizer = Adam(learning_rate=learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # Build shapes safely
        dummy_input = tf.zeros((1, max_seq_len - 1), dtype=tf.int32)
        _ = model(dummy_input)

        # Summary Table
        st.subheader("📋 Architectural & Parameter Breakdown")
        summary_data = []
        for layer in model.layers:
            try:
                out_shape = str(layer.output_shape)
            except AttributeError:
                out_shape = str(layer.compute_output_shape((None, max_seq_len - 1)))
            summary_data.append({
                "Layer Name": layer.name,
                "Layer Type": layer.__class__.__name__,
                "Output Shape": out_shape,
                "Param #": f"{layer.count_params():,}"
            })
        st.table(pd.DataFrame(summary_data))

        col1, col2 = st.columns(2)
        col1.metric("Total Parameters", f"{model.count_params():,}")
        col2.metric("Trainable Parameters", f"{sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")

        # Training Monitor
        st.subheader("Active Training Monitor")
        log_placeholder = st.empty()
        callback = StreamlitTrainingCallback(log_placeholder, epochs)

        with st.spinner("Training model..."):
            model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=[callback], verbose=0)

        # Save session state
        st.session_state['trained_model'] = model
        st.session_state['tokenizer'] = tokenizer
        st.session_state['max_seq_len'] = max_seq_len
        st.success("🎉 Model trained successfully! Switch to the **Inference** tab to test predictions.")


# --- TAB 2: INFERENCE & MULTI-WORD PREDICTION ---
with tab2:
    st.header("Generate Predictions")
    
    if 'trained_model' not in st.session_state:
        st.warning("⚠️ No trained model found! Please train a model in Tab 1 first.")
    else:
        model = st.session_state['trained_model']
        tokenizer = st.session_state['tokenizer']
        max_seq_len = st.session_state['max_seq_len']

        seed_text = st.text_input("Enter Seed Text / Prompt:", "Language modeling is")
        next_words = st.slider("Number of Words to Predict:", min_value=1, max_value=10, value=3)

        if st.button("🔮 Predict Next Words"):
            current_text = seed_text
            predicted_words = []

            for _ in range(next_words):
                token_list = tokenizer.texts_to_sequences([current_text])[0]
                token_list = pad_sequences([token_list], maxlen=max_seq_len - 1, padding='pre')
                predicted_probs = model.predict(token_list, verbose=0)[0]
                predicted_index = np.argmax(predicted_probs)

                output_word = ""
                for word, index in tokenizer.word_index.items():
                    if index == predicted_index:
                        output_word = word
                        break

                if not output_word:
                    break

                current_text += " " + output_word
                predicted_words.append(output_word)

            # --- DISPLAY RESULTS WITH CSS STYLING ---
            st.subheader("Generated Output:")
            
            st.markdown(f"""
                <div class="prediction-box">
                    <p class="prompt-label">SEED PROMPT</p>
                    <p class="prompt-text">"{seed_text}"</p>
                    <hr style="border: 0; border-top: 1px solid #444; margin: 15px 0;">
                    <p class="prompt-label">PREDICTED SEQUENCE</p>
                    <p class="result-text">{current_text}</p>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("### Predicted Words List")
            badges_html = " ".join([
                f'<span style="background-color: #3b82f6; color: white; padding: 4px 12px; border-radius: 15px; font-weight: 600; font-size: 0.9rem; display: inline-block; margin: 3px;">{word}</span>'
                for word in predicted_words
            ])
            st.markdown(badges_html, unsafe_allow_html=True)
