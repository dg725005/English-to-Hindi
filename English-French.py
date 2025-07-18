import tensorflow as tf

from transformers import AutoTokenizer
from transformers import TFAutoModelForSeq2SeqLM # Its purpose is to automatically load a pre-trained 
                                                 # sequence-to-sequence language model that is 
                                                 # compatible with TensorFlow.


print(f"TensorFlow version: {tf.__version__}") # Print the TensorFlow version for verification.

MODEL_NAME = "Helsinki-NLP/opus-mt-en-hi"

import sentencepiece
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load the tokenizer associated with the pre-trained model.
# The tokenizer is responsible for converting text into numerical IDs (tokens)

model = TFAutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Load the pre-trained sequence-to-sequence model for conditional generation (translation).
# TFAutoModelForSeq2SeqLM automatically selects the correct model architecture
# (e.g., MarianMT model in this case) and loads its pre-trained weights, ensuring
# it's compatible with TensorFlow. This model has an encoder (BERT-like)
# and a decoder (GPT-like) component.

def translate_text(text_to_translate):
    inputs = tokenizer(text_to_translate, return_tensors="tf", truncation=True,padding=True)

    # Generate the translation using the model.
    # 'generate' method is used for sequence generation tasks.
    # 'inputs' are the tokenized input IDs.
    # 'max_length' sets the maximum length of the generated output sequence.
    # 'num_beams' is used for beam search decoding, which explores multiple
    # possible next words to find a more probable sequence, leading to better translations.
    # 'early_stopping=True' stops generation once all beam hypotheses have finished.
    # 'no_repeat_ngram_size=2' prevents the generation of repeating n-grams (e.g., words or phrases)
    # of size 2 or more, which helps in producing more natural-sounding translations.

    translated_tokens = model.generate(
        inputs["input_ids"],
        max_length = 50,
        num_beams = 5,
        early_stopping = True,
        no_repeat_ngram_size = 2
    )

    # Decode the generated tokens back into human-readable text.
    # 'skip_special_tokens=True' removes special tokens (like [CLS], [SEP], [PAD])
    # from the decoded output, resulting in clean text.
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

    # Return the translated text.
    return translated_text

import streamlit as st

T = st.text_input("Enter text to translate:")
if T:
    translated_text = translate_text(T)
    st.write(f'Input Text: {T}')
    st.write(f'Translated Text: {translated_text}')