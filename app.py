import torch
import torch.nn as nn
import streamlit as st
import pickle

# Load vocabulary
with open('simple_vocab (2).pkl', 'rb') as f:
    vocab = pickle.load(f)

class PoetryLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2):
        super(PoetryLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)
        out = self.fc(lstm_out[:, -1])
        return out

# Load Model
model = PoetryLSTM(vocab_size=len(vocab.stoi))
model.load_state_dict(torch.load('poetry_lstm_model.pth', map_location=torch.device('cpu')))
model.eval()

def generate_poetry(seed_text, model, vocab, max_words=68):
    words = seed_text.split()
    for _ in range(max_words):
        encoded = torch.tensor([vocab[word] for word in words[-6:]]).unsqueeze(0)
        with torch.no_grad():
            output = model(encoded)
            next_word = vocab.lookup_token(output.argmax().item())
            words.append(next_word)
    return " ".join(words)

def print_poetry(generated_text):
    return generated_text.replace(" <NEWLINE> ", "\n").replace(" <NEWLINE>", "\n")

# Streamlit App Styling
st.markdown(
    """
    <style>
    body {
        background-color: black;
        color: gold;
        font-family: 'Courier New', monospace;
    }
    .stTextInput, .stNumberInput, .stButton {
        background-color: #222;
        color: gold;
        border-radius: 10px;
    }
    .stButton > button {
        background-color: gold !important;
        color: black !important;
        font-weight: bold;
    }
    .poetry-box {
        background-color: #111;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title('Shayri AI ✨')
input_text = st.text_input('Enter a word to generate poetry:', '')
num_words = st.number_input('Number of words:', min_value=5, max_value=100, value=20)
if st.button('Generate Poetry'):
    if input_text:
        generated_poetry = generate_poetry(input_text, model, vocab, num_words)
        poetry = print_poetry(generated_poetry)
        st.markdown(f'<div class="poetry-box">{poetry.replace("\n", "<br>")}</div>', unsafe_allow_html=True)
    else:
        st.write("Please enter some words to generate poetry.")
