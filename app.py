import streamlit as st
import torch
import pandas as pd

def load_css(css_file_path):
    with open(css_file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css('style.css')


class MinimalTransformer(torch.nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, forward_expansion, gender_size):
        super(MinimalTransformer, self).__init__()
        self.embed = torch.nn.Embedding(vocab_size, embed_size)
        self.gender_embed = torch.nn.Embedding(gender_size, embed_size)
        self.positional_encoding = torch.nn.Parameter(torch.randn(1, 100, embed_size))
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads)
        self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.output_layer = torch.nn.Linear(embed_size, vocab_size)

    def forward(self, x, gender):
        gender_emb = self.gender_embed(gender).unsqueeze(1).expand(-1, x.size(1), -1)
        x = self.embed(x) + self.positional_encoding[:, :x.size(1), :] + gender_emb
        x = self.transformer_encoder(x)
        x = self.output_layer(x)
        return x


class NameDataset:
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)
        self.names = data['name'].values
        self.genders = data['gender'].values

        self.chars = sorted(list(set(''.join(self.names) + ' ')))
        self.char_to_int = {c: i for i, c in enumerate(self.chars)}
        self.int_to_char = {i: c for c, i in self.char_to_int.items()}
        self.vocab_size = len(self.chars)

        self.gender_to_int = {'male': 0, 'female': 1}
        self.int_to_gender = {0: 'male', 1: 'female'}


# Load the dataset and model
@st.cache_resource
def load_resources():
    dataset = NameDataset('names_dataset.csv')
    model = torch.load('namesformer_model.pt', map_location=torch.device('cpu'))
    model.eval()
    return model, dataset


# Sampling function
def sample(model, dataset, start_str='a', max_length=20, temperature=1.0, gender='male'):
    assert temperature > 0, "Temperature must be greater than 0"
    with torch.no_grad():
        chars = [dataset.char_to_int[c] for c in start_str]
        input_seq = torch.tensor(chars).unsqueeze(0)
        gender_tensor = torch.tensor([dataset.gender_to_int[gender]])

        output_name = start_str
        for _ in range(max_length - len(start_str)):
            output = model(input_seq, gender_tensor)
            logits = output[0, -1] / temperature
            probabilities = torch.softmax(logits, dim=0)
            next_char_idx = torch.multinomial(probabilities, 1).item()
            next_char = dataset.int_to_char[next_char_idx]

            if next_char == ' ':
                break

            output_name += next_char
            input_seq = torch.cat([input_seq, torch.tensor([[next_char_idx]])], dim=1)

        return output_name


# Interface
st.title("Lietuviškų vardų generatorius")
st.write("Ar jums trūksta idėjų? Ar norite unikalaus vardo, kurio neturėtų niekas Lietuvoje? Leiskite dirbtiniam intelektui sugalvoti jums vardą!")


# Inputs
start_str = st.text_input("Vardo pradžia:", "A")
gender_input = st.selectbox("Lytis:", ["berniukas", "mergaitė"])

# Fixing issues due to localized gender :D
gender_translation = {
    'berniukas': 'male',
    'mergaitė': 'female'
}
gender = gender_translation[gender_input]

model, dataset = load_resources()

# Generates name
if st.button("Kurti vardą"):
    generated_name = sample(model, dataset, start_str=start_str, max_length=20, temperature=0.5, gender=gender)
    st.success(f"Jūsų vardas: {generated_name}")

# Generates a very creative name
if st.button("Aš influenceris"):
    generated_name = sample(model, dataset, start_str=start_str, max_length=20, temperature=2.0, gender=gender)
    st.success(f"Jūsų vardas: {generated_name}")

    st.markdown("""
        <script>
        const influencerButton = [...document.querySelectorAll('.stButton button')]
          .find(el => el.innerText === "Aš influenceris");
        influencerButton.addEventListener("click", function() {
            confetti({
                particleCount: 100,
                spread: 70,
                origin: { y: 0.6 }
            });
        });
        </script>
        <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.6.0/dist/confetti.browser.min.js"></script>
    """, unsafe_allow_html=True)
