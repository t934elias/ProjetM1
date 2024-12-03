from transformers import MarianMTModel, MarianTokenizer
import pandas as pd
import os

data_path = os.path.join(os.path.dirname(__file__), "datasets/data/DAIC-WOZ")
data_fr_path = os.path.join(os.path.dirname(__file__), "datasets/data_fr/DAIC-WOZ_FR")
data = pd.read_csv(os.path.join(data_path, "300_TRANSCRIPT.csv"), sep="\t")

texts = data["value"]
# print(texts.isna().sum())

data_fr = pd.DataFrame()
data_fr = data.copy(deep=True)
# data_fr['original'] = texts
data_fr['trad'] = "Francais"
# print(data_fr.head())

# Nom du modèle pour la traduction EN->FR
model_name = "Helsinki-NLP/opus-mt-en-fr"

# Charger le tokenizer et le modèle
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Traduire les textes
trad_tab = []
for text in texts:
    tokenized_text = tokenizer(text, return_tensors="pt", padding=True)
    translated = model.generate(**tokenized_text)
    trad_tab.append(tokenizer.decode(translated[0], skip_special_tokens=True))
data_fr['trad'] = trad_tab

# Sauvegarder les traductions
data_fr.to_csv(os.path.join(data_fr_path, "300_TRANSCRIPT_FR.csv"), sep="\t", index=False)
print("Traduction terminée")