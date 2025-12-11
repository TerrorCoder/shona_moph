# Save this as 'fortune_grammar.py' in your folder
# Based on George Fortune's "Shona Grammatical Constructions"

SHONA_CLASS_MAP = {
    "mu": [
        {"class": "1", "meaning": "Person", "number": "Singular", "plural_prefix": "va"},
        {"class": "3", "meaning": "Tree/Atmospheric", "number": "Singular", "plural_prefix": "mi"},
        {"class": "18", "meaning": "Locative (Inside)", "number": "N/A", "plural_prefix": None}
    ],
    "va": [
        {"class": "2", "meaning": "People (Plural of 1)", "number": "Plural", "singular_prefix": "mu"},
        {"class": "2a", "meaning": "Honorific/Respect", "number": "N/A", "singular_prefix": None}
    ],
    "mi": [
        {"class": "4", "meaning": "Trees/Miscellaneous (Plural of 3)", "number": "Plural", "singular_prefix": "mu"}
    ],
    "chi": [
        {"class": "7", "meaning": "Object/Language/Short Person", "number": "Singular", "plural_prefix": "zvi"}
    ],
    "zvi": [
        {"class": "8", "meaning": "Objects (Plural of 7)", "number": "Plural", "singular_prefix": "chi"}
    ],
    "ma": [
        {"class": "6", "meaning": "Liquids/Plurals of Cl 5", "number": "Plural", "singular_prefix": "ri"}
    ],
    "ri": [
        {"class": "5", "meaning": "Large Object/Fruit", "number": "Singular", "plural_prefix": "ma"}
    ],
    "ru": [
        {"class": "11", "meaning": "Long/Thin Object or Abstract", "number": "Singular", "plural_prefix": "n"} # or ma-
    ],
    "ka": [
        {"class": "12", "meaning": "Diminutive (Small)", "number": "Singular", "plural_prefix": "tu"}
    ],
    "tu": [
        {"class": "13", "meaning": "Diminutive (Plural of 12)", "number": "Plural", "singular_prefix": "ka"}
    ],
    "hu": [
        {"class": "14", "meaning": "Abstract Quality", "number": "Abstract", "plural_prefix": None}
    ],
    "ku": [
        {"class": "15", "meaning": "Infinitive (To do)", "number": "N/A", "plural_prefix": None},
        {"class": "17", "meaning": "Locative (At/To)", "number": "N/A", "plural_prefix": None}
    ],
    "pa": [
        {"class": "16", "meaning": "Locative (At/On)", "number": "N/A", "plural_prefix": None}
    ]
}

def analyze_morphology(prefix, stem):
    prefix = prefix.lower()
    if prefix in SHONA_CLASS_MAP:
        candidates = SHONA_CLASS_MAP[prefix]
        # Heuristic: If multiple candidates, return all or try to guess?
        # For now, we return all possibilities.
        return candidates
    else:
        # Fallback for Class 1a, 5 (Zero prefix), 9 (Nasal)
        # This usually means the AI returned the whole word as stem or "No Prefix"
        return [{"class": "Unknown (1a, 5, 9)", "meaning": "No visible prefix", "number": "Singular", "plural_prefix": "Unknown"}]

def get_lemma(prefix, stem, analysis_list):
    # Try to generate the Singular form (Lemma)
    lemmas = []
    for item in analysis_list:
        if item.get("number") == "Plural" and item.get("singular_prefix"):
            # Replace plural prefix with singular prefix
            # e.g. zvi-bage -> chi-bage
            lemmas.append(item["singular_prefix"] + stem)
        elif item.get("number") == "Singular":
            # Lemma is the word itself
            lemmas.append(prefix + stem)
        else:
            lemmas.append(prefix + stem)
    return list(set(lemmas)) # Unique lemmas