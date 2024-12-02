from transformers import MarianMTModel, MarianTokenizer

# Nom du modèle pour la traduction EN->FR
model_name = "Helsinki-NLP/opus-mt-en-fr"

# Charger le tokenizer et le modèle
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Exemple de texte à traduire
text = "Hello, how are you today?"
# Tokenisation
tokenized_text = tokenizer(text, return_tensors="pt", padding=True)

# Traduction
translated = model.generate(**tokenized_text)

# Décoder le texte traduit
translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
print("Traduction :", translated_text)