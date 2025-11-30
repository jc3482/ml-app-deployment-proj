import json
import os
from rapidfuzz import process, fuzz


class YoloPreprocessor:
    """
    Preprocess YOLO class labels into canonical ingredient vocabulary.

    Steps:
    1. Normalize label string (underscore → space, lowercase, strip)
    2. Exact match check
    3. Fuzzy match using RapidFuzz extractOne()
       - extractOne returns (match, score, index) → MUST unpack correctly
    4. Fallback matching using first token ("chicken breast" → "chicken")
    5. Return None for items with no meaningful match
    """

    def __init__(self, vocab_path="data/canonical_vocab.json", fuzzy_threshold=80):
        # Resolve project root path so this works no matter where script is run
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        vocab_path = os.path.join(base_dir, vocab_path)

        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"canonical_vocab.json not found at {vocab_path}")

        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = [v.lower().strip() for v in json.load(f)]

        self.threshold = fuzzy_threshold

    # --------------------------------------------------------
    # Normalize one YOLO label
    # --------------------------------------------------------
    def normalize_single(self, label: str):
        """
        Normalize one YOLO detection label into a canonical ingredient.
        Returns None if no good match exists.
        """
        if not isinstance(label, str):
            return None

        raw = label.replace("_", " ").lower().strip()

        # 1. Exact match
        if raw in self.vocab:
            return raw

        # 2. Fuzzy match with RapidFuzz
        result = process.extractOne(raw, self.vocab, scorer=fuzz.ratio)

        if result is not None:
            match, score, _ = result  # RapidFuzz → (best_match, score, index)
            if score >= self.threshold:
                return match

        # 3. Try fallback: use first word if multi-word
        tokens = raw.split()
        if len(tokens) > 1:
            base = tokens[0]
            sub_result = process.extractOne(base, self.vocab, scorer=fuzz.ratio)
            if sub_result is not None:
                sub_match, sub_score, _ = sub_result
                if sub_score >= self.threshold:
                    return sub_match

        # 4. No valid match
        return None

    # --------------------------------------------------------
    # Normalize list of YOLO outputs
    # --------------------------------------------------------
    def normalize(self, labels):
        """
        Normalize a list of YOLO outputs into canonical ingredients.
        Deduplicates and sorts results.
        """
        norm = []

        for l in labels:
            fixed = self.normalize_single(l)
            if fixed:
                norm.append(fixed)

        # unique & sorted
        return sorted(set(norm))
