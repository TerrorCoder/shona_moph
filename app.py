import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import mysql.connector

# ------------------------------------------------
# 1. MYSQL DATABASE CONNECTION
# ------------------------------------------------
def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",  # XAMPP default
        database="shona_ai"
    )

def save_to_db(word, prefix, stem, full_split, class_id, meaning):
    conn = get_connection()
    cursor = conn.cursor()

    sql = """
        INSERT INTO analysis_log (word, prefix, stem, full_split, predicted_class, meaning)
        VALUES (%s, %s, %s, %s, %s, %s)
    """
    values = (word, prefix, stem, full_split, class_id, meaning)

    cursor.execute(sql, values)
    conn.commit()

    cursor.close()
    conn.close()

# ------------------------------------------------
# 2. SHONA RULE-BASED DICTIONARY (FORTUNE)
# ------------------------------------------------
SHONA_CLASS_MAP = {
    "mu": [
        {"class": "1", "meaning": "Person/agent/people who undergo action indicated by agent", "number": "Singular", "plural": "va", "priority": 1},
        {"class": "3", "meaning": "Tree/Atmospheric/body parts/manner of action", "number": "Singular", "plural": "mi", "priority": 2},
        {"class": "18", "meaning": "Locative (Inside)", "number": "N/A", "plural": None, "priority": 3}
    ],
    "va": [
        {"class": "2", "meaning": "People", "number": "Plural", "lemma_prefix": "mu", "priority": 1},
        {"class": "2a", "meaning": "Honorific", "number": "N/A", "lemma_prefix": None, "priority": 2}
    ],
    "mi": [
        {"class": "4", "meaning": "Trees/Misc", "number": "Plural", "lemma_prefix": "mu", "priority": 1}
    ],
    "chi": [
        {"class": "7", "meaning": "Object/Lang", "number": "Singular", "plural": "zvi", "priority": 1}
    ],
    "zvi": [
        {"class": "8", "meaning": "Objects", "number": "Plural", "lemma_prefix": "chi", "priority": 1}
    ],
    "ma": [
        {"class": "6", "meaning": "Liquids/Big Things", "number": "Plural", "lemma_prefix": "ri", "priority": 1}
    ],
    "ri": [
        {"class": "5", "meaning": "Large Object/Fruit", "number": "Singular", "plural": "ma", "priority": 1}
    ],
    "ru": [
        {"class": "11", "meaning": "Long/Thin Object", "number": "Singular", "plural": "n", "priority": 1}
    ],
    "ka": [
        {"class": "12", "meaning": "Diminutive (Small)", "number": "Singular", "plural": "tu", "priority": 1}
    ],
    "tu": [
        {"class": "13", "meaning": "Diminutive Plural", "number": "Plural", "lemma_prefix": "ka", "priority": 1}
    ],
    "hu": [
        {"class": "14", "meaning": "Abstract Quality", "number": "Abstract", "plural": None, "priority": 1}
    ],
    "ku": [
        {"class": "15", "meaning": "Infinitive (To do)", "number": "N/A", "plural": None, "priority": 1},
        {"class": "17", "meaning": "Locative (At/To)", "number": "N/A", "plural": None, "priority": 2}
    ],
    "pa": [
        {"class": "16", "meaning": "Locative (At/On)", "number": "N/A", "plural": None, "priority": 1}
    ],
    "svi": [
        {"class": "19", "meaning": "Diminutive/Pejorative (Small/Bad)", "number": "Singular", "plural": None, "priority": 1}
    ],
    "zi": [
        {"class": "21", "meaning": "Augmentative/Pejorative (Large/Excessive)", "number": "Singular", "plural": None, "priority": 1}
    ]
}

# Disambiguation: Common stem patterns
PERSON_STEMS = ["nhu", "ntu", "kuru", "rume", "fazi", "ana", "komana", "sikana"]
LOCATIVE_STEMS = ["munda", "minda", "musha", "rodhi", "gomo", "dziva"]  # Add location-related stems
TREE_STEMS = ["ti", "tondo", "pani", "ndo", "sasa", "tsamvu", "nzviro"]
BODY_PART_STEMS = ["soro", "romo", "mhuno", "romo", "dzira"]

def select_best_class(prefix, stem, candidates):
    """
    Select the most appropriate noun class from multiple candidates
    """
    if len(candidates) == 1:
        return candidates[0]
    
    stem_lower = stem.lower()
    
    # Special handling for "mu" prefix
    if prefix == "mu":
        # Check for person stems
        if any(stem_lower.startswith(s) or stem_lower == s for s in PERSON_STEMS):
            return next(c for c in candidates if c["class"] == "1")
        
    # Class 18 locatives: mu + [location noun]
        if any(stem_lower.startswith(s) or stem_lower == s for s in LOCATIVE_STEMS):
            # Return class 18 if it exists among candidates
            for c in candidates:
                if c["class"] == "18":
                 return c
            
        # Check for tree stems
        if any(stem_lower.startswith(s) or stem_lower == s for s in TREE_STEMS + BODY_PART_STEMS):
            return next(c for c in candidates if c["class"] == "3")
        
        # Default to Class 1 (most common)
        return candidates[0]
    
    # For "ku" prefix, prefer infinitive (Class 15)
    if prefix == "ku":
        return candidates[0]
    
    # For other cases, return highest priority
    return min(candidates, key=lambda x: x.get("priority", 99))

# ------------------------------------------------
# 3. LOAD MODEL + TOKENIZER
# ------------------------------------------------
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('shona_morphology_final.keras')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

model, tokenizer = load_assets()
MAX_LEN = 30

# ------------------------------------------------
# 4. AI PREDICTION FUNCTION
# ------------------------------------------------
def ai_predict_split(word):
    seq = tokenizer.texts_to_sequences([word.lower()])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    pred = model.predict(padded, verbose=0)[0]

    split_word = ""
    prefix = ""
    stem = ""
    split_found = False
    
    for i, char in enumerate(word):
        if i < len(pred) and pred[i] > 0.5 and i < len(word) - 1 and not split_found:
            split_word += char + "-"
            prefix += char
            split_found = True
        else:
            split_word += char
            if split_found:
                stem += char
            else:
                prefix += char

    if not split_found:
        return word, "", word

    return split_word, prefix, stem

# ------------------------------------------------
# 5. STREAMLIT UI
# ------------------------------------------------
st.set_page_config(page_title="Shona Analyser", page_icon="zw")

st.title(" Deep Shona Morphological Analyser")
st.markdown("AI + Fortune Grammar Hybrid Morphology System")

word_input = st.text_input("Enter a Shona Word:", value="munhu")

if st.button("Deep Analysis"):
    if word_input:
        full_split, prefix, stem = ai_predict_split(word_input)
        analysis_data = SHONA_CLASS_MAP.get(prefix.lower(), [])

        st.subheader(f"Segmentation: {full_split}")
        st.metric("Prefix", prefix if prefix else "Ã˜ (Zero Prefix)")
        st.metric("Stem", f"-{stem}")

        st.divider()
        st.write("### Grammatical Analysis")

        if analysis_data:
            # SELECT ONLY ONE BEST CLASS
            best_candidate = select_best_class(prefix, stem, analysis_data)

            # Save to database
            save_to_db(
                word_input,
                prefix,
                stem,
                full_split,
                best_candidate["class"],
                best_candidate["meaning"]
            )
            st.success("Saved to database ¸")

            # Calculate lemma
            lemma = word_input
            if best_candidate.get("number") == "Plural" and best_candidate.get("lemma_prefix"):
                lemma = best_candidate["lemma_prefix"] + stem

            # Display ONLY the best candidate
            with st.expander(
                f"Candidate: Class {best_candidate['class']} ({best_candidate['meaning']})",
                expanded=True
            ):
                st.write(f"**Class:** {best_candidate['class']}")
                st.write(f"**Number:** {best_candidate['number']}")
                st.write(f"**Meaning:** {best_candidate['meaning']}")
                st.write(f"**Lemma:** {lemma}")
                if best_candidate.get("plural"):
                    st.write(f"**Plural Form:** {best_candidate['plural']}{stem}")

        else:
            save_to_db(
                word_input,
                prefix,
                stem,
                full_split,
                "unknown",
                "unknown"
            )
            st.warning("Prefix not found in rule book. Saved as unknown class.")

    else:
        st.error("Please enter a word.")