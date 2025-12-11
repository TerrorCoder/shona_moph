import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- 1. FORTUNE GRAMMAR LOGIC (Pasted here for easy running) ---
SHONA_CLASS_MAP = {
    "mu": [
        {"class": "1", "meaning": "Person", "number": "Singular", "plural": "va"},
        {"class": "3", "meaning": "Tree/Atmospheric", "number": "Singular", "plural": "mi"},
        {"class": "18", "meaning": "Locative (Inside)", "number": "N/A", "plural": None}
    ],
    "va": [
        {"class": "2", "meaning": "People", "number": "Plural", "lemma_prefix": "mu"},
        {"class": "2a", "meaning": "Honorific", "number": "N/A", "lemma_prefix": None}
    ],
    "mi": [
        {"class": "4", "meaning": "Trees/Misc", "number": "Plural", "lemma_prefix": "mu"}
    ],
    "chi": [
        {"class": "7", "meaning": "Object/Lang", "number": "Singular", "plural": "zvi"}
    ],
    "zvi": [
        {"class": "8", "meaning": "Objects", "number": "Plural", "lemma_prefix": "chi"}
    ],
    "ma": [
        {"class": "6", "meaning": "Liquids/Big Things", "number": "Plural", "lemma_prefix": "ri"} # or Ø
    ],
    # ... Add other prefixes (ru, ka, tu, hu, ku, pa) here as needed
}

# --- 2. LOAD AI MODEL ---
@st.cache_resource
def load_assets():
    # Load your model and tokenizer
    model = tf.keras.models.load_model('shona_morphology_final.keras')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

model, tokenizer = load_assets()
MAX_LEN = 30 

# --- 3. AI PREDICTION FUNCTION ---
def ai_predict_split(word):
    seq = tokenizer.texts_to_sequences([word.lower()])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    pred = model.predict(padded, verbose=0)[0]
    
    # Reconstruct split
    split_word = ""
    prefix = ""
    stem = ""
    split_found = False
    
    for i, char in enumerate(word):
        # Check for split
        if i < len(pred) and pred[i] > 0.5 and i < len(word) - 1 and not split_found:
            split_word += char + "-"
            prefix += char
            split_found = True # Only allow one main split for prefix
        else:
            split_word += char
            if split_found:
                stem += char
            else:
                prefix += char
                
    if not split_found:
        return word, "", word # No split (Prefix is empty or zero)
    
    return split_word, prefix, stem

# --- 4. UI DISPLAY ---
st.set_page_config(page_title="Shona Analyser", page_icon="🇿🇼")

st.title("🇿🇼 Deep Shona Morphological Analyser")
st.markdown("Combines **BiLSTM Neural Networks** with **George Fortune's Grammatical Rules**.")

word_input = st.text_input("Enter a Shona Word:", value="zvibage")

if st.button("Deep Analysis"):
    if word_input:
        # A. Run AI Splitter
        full_split, prefix, stem = ai_predict_split(word_input)
        
        # B. Run Fortune Logic
        analysis_data = SHONA_CLASS_MAP.get(prefix.lower(), [])
        
        # C. Display Results
        st.subheader(f"Segmentation: {full_split}")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Prefix", prefix if prefix else "Ø (Zero)")
        col2.metric("Stem", f"-{stem}")
        st.divider()
        st.write("### Grammatical Analysis")

        if analysis_data:
            # Select only the first/most likely candidate
            candidate = analysis_data[0]
            
            # Calculate Lemma
            lemma = word_input # Default
            if candidate.get('number') == 'Plural' and candidate.get('lemma_prefix'):
                lemma = candidate['lemma_prefix'] + stem
                
            st.write(f"**Class:** {candidate['class']}")
            st.write(f"**Meaning:** {candidate['meaning']}")
            st.write(f"**Number:** {candidate['number']}")
            st.write(f"**Lemma (Root/Singular):** {lemma}")
            if candidate.get('plural'):
                st.write(f"**Plural Form:** {candidate['plural']}{stem}")
        else:
            st.warning("Prefix not found in Rule Book (or it is a Zero Prefix noun like Class 5/9).")
            st.write("Possible Classes: **1a** (Relationships), **5** (ri-), **9** (N-), or **10**.")
    else:
        st.error("Please enter a word.")