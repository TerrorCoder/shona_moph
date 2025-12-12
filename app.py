import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from datetime import datetime
import json

# Optional Google Sheets integration
try:
    import gspread
    from google.oauth2.service_account import Credentials
    SHEETS_AVAILABLE = True
except ImportError:
    SHEETS_AVAILABLE = False

# ------------------------------------------------
# 1. GOOGLE SHEETS SETUP (Optional)
# ------------------------------------------------
def init_google_sheets():
    """Initialize Google Sheets connection"""
    if not SHEETS_AVAILABLE:
        return None
    
    try:
        # Get credentials from Streamlit secrets
        creds_dict = st.secrets.get("gcp_service_account", None)
        if not creds_dict:
            return None
        
        # Setup credentials
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        client = gspread.authorize(creds)
        
        # Open or create spreadsheet
        sheet_name = st.secrets.get("sheet_name", "Shona Analysis Log")
        try:
            spreadsheet = client.open(sheet_name)
        except gspread.SpreadsheetNotFound:
            spreadsheet = client.create(sheet_name)
            spreadsheet.share('', perm_type='anyone', role='reader')
        
        worksheet = spreadsheet.sheet1
        
        # Setup headers if empty
        if worksheet.row_count == 0 or worksheet.acell('A1').value != 'Timestamp':
            worksheet.update('A1:G1', [['Timestamp', 'Word', 'Prefix', 'Stem', 'Full Split', 'Class', 'Meaning']])
        
        return worksheet
    except Exception as e:
        st.sidebar.error(f"Google Sheets error: {e}")
        return None

def save_to_sheets(worksheet, word, prefix, stem, full_split, class_id, meaning):
    """Save analysis to Google Sheets"""
    try:
        row = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            word,
            prefix,
            stem,
            full_split,
            class_id,
            meaning
        ]
        worksheet.append_row(row)
        return True
    except Exception as e:
        st.error(f"Failed to save to Google Sheets: {e}")
        return False

# ------------------------------------------------
# 2. SESSION STATE FOR LOCAL HISTORY
# ------------------------------------------------
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

def add_to_history(word, prefix, stem, full_split, class_id, meaning):
    """Add analysis to session history"""
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "word": word,
        "prefix": prefix,
        "stem": stem,
        "full_split": full_split,
        "class": class_id,
        "meaning": meaning
    }
    st.session_state.analysis_history.insert(0, entry)  # Add to beginning
    
    # Keep only last 100 entries
    if len(st.session_state.analysis_history) > 100:
        st.session_state.analysis_history = st.session_state.analysis_history[:100]

def get_history_df():
    """Convert history to DataFrame"""
    if not st.session_state.analysis_history:
        return pd.DataFrame()
    return pd.DataFrame(st.session_state.analysis_history)

# ------------------------------------------------
# 3. SHONA RULE-BASED DICTIONARY (FORTUNE)
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
LOCATIVE_STEMS = ["munda", "minda", "musha", "rodhi", "gomo", "dziva"]
TREE_STEMS = ["ti", "tondo", "pani", "ndo", "sasa", "tsamvu", "nzviro"]
BODY_PART_STEMS = ["soro", "romo", "mhuno", "romo", "dzira"]

def select_best_class(prefix, stem, candidates):
    """Select the most appropriate noun class from multiple candidates"""
    if len(candidates) == 1:
        return candidates[0]
    
    stem_lower = stem.lower()
    
    # Special handling for "mu" prefix
    if prefix == "mu":
        # Class 18 locatives first
        if any(stem_lower.startswith(s) or stem_lower == s for s in LOCATIVE_STEMS):
            for c in candidates:
                if c["class"] == "18":
                    return c
        
        # Check for person stems
        if any(stem_lower.startswith(s) or stem_lower == s for s in PERSON_STEMS):
            return next(c for c in candidates if c["class"] == "1")
        
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
# 4. LOAD MODEL + TOKENIZER
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
# 5. AI PREDICTION FUNCTION
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
# 6. STREAMLIT UI
# ------------------------------------------------
st.set_page_config(page_title="Shona Analyser", page_icon="🇿🇼")

st.title("🇿🇼 Deep Shona Morphological Analyser")
st.markdown("AI + Fortune Grammar Hybrid Morphology System")

# Sidebar for Google Sheets status
with st.sidebar:
    st.header("Settings")
    
    if SHEETS_AVAILABLE:
        worksheet = init_google_sheets()
        if worksheet:
            st.success("✅ Google Sheets connected")
            enable_sheets = st.checkbox("Save to Google Sheets", value=True)
        else:
            st.warning("⚠️ Google Sheets not configured")
            st.info("Add Google Sheets credentials to Streamlit secrets to enable cloud storage.")
            enable_sheets = False
    else:
        st.info("📦 Install gspread for Google Sheets support:\n```pip install gspread google-auth```")
        worksheet = None
        enable_sheets = False
    
    st.divider()
    
    # Download history button
    if st.session_state.analysis_history:
        st.subheader("Export Data")
        df = get_history_df()
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download History (CSV)",
            data=csv,
            file_name=f"shona_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        if st.button("🗑️ Clear History"):
            st.session_state.analysis_history = []
            st.rerun()

# Tabs for analysis and history
tab1, tab2 = st.tabs(["Analyze Word", "Analysis History"])

with tab1:
    word_input = st.text_input("Enter a Shona Word in lowercase:", value="munhu")

    if st.button("🔍 Deep Analysis", type="primary"):
        if word_input:
            full_split, prefix, stem = ai_predict_split(word_input)
            analysis_data = SHONA_CLASS_MAP.get(prefix.lower(), [])

            st.subheader(f"Segmentation: {full_split}")
            
            col1, col2 = st.columns(2)
            col1.metric("Prefix", prefix if prefix else "Ø (Zero)")
            col2.metric("Stem", f"-{stem}")

            st.divider()
            st.write("### Grammatical Analysis")

            if analysis_data:
                # SELECT ONLY ONE BEST CLASS
                best_candidate = select_best_class(prefix, stem, analysis_data)

                # Calculate lemma
                lemma = word_input
                if best_candidate.get("number") == "Plural" and best_candidate.get("lemma_prefix"):
                    lemma = best_candidate["lemma_prefix"] + stem

                # Display ONLY the best candidate
                with st.container():
                    st.write(f"**Class:** {best_candidate['class']}")
                    st.write(f"**Number:** {best_candidate['number']}")
                    st.write(f"**Meaning:** {best_candidate['meaning']}")
                    st.write(f"**Lemma:** {lemma}")
                    if best_candidate.get("plural"):
                        st.write(f"**Plural Form:** {best_candidate['plural']}{stem}")
                    elif best_candidate["class"] == "18":
                        if stem == "munda":
                            st.write(f"**Plural Form:** muminda")
                        # Add other locative plural rules as needed

                # Save to history
                add_to_history(word_input, prefix, stem, full_split, best_candidate['class'], best_candidate['meaning'])
                
                # Save to Google Sheets if enabled
                if enable_sheets and worksheet:
                    if save_to_sheets(worksheet, word_input, prefix, stem, full_split, best_candidate['class'], best_candidate['meaning']):
                        st.success("💾 Saved to Google Sheets")

            else:
                st.warning("Prefix not found in rule book (or it is a Zero Prefix noun like Class 5/9).")
                st.write("Possible Classes: **1a** (Relationships), **5** (ri-), **9** (N-), or **10**.")
                
                # Save unknown to history
                add_to_history(word_input, prefix, stem, full_split, "unknown", "unknown")
                
                if enable_sheets and worksheet:
                    save_to_sheets(worksheet, word_input, prefix, stem, full_split, "unknown", "unknown")

        else:
            st.error("Please enter a word.")

with tab2:
    st.write("### Recent Analyses")
    
    if st.session_state.analysis_history:
        df = get_history_df()
        
        # Display stats
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Analyses", len(df))
        col2.metric("Unique Words", df['word'].nunique())
        col3.metric("Unique Classes", df['class'].nunique())
        
        st.divider()
        
        # Display dataframe
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )
        
        # Quick search
        st.subheader("Search History")
        search_term = st.text_input("Search for a word:")
        if search_term:
            filtered = df[df['word'].str.contains(search_term, case=False, na=False)]
            if not filtered.empty:
                st.dataframe(filtered, use_container_width=True, hide_index=True)
            else:
                st.info("No matches found.")
    else:
        st.info("No analyses yet. Start analyzing words in the 'Analyze Word' tab!")