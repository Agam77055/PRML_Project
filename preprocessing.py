import re
import numpy as np
import wn
from collections import Counter

class PreProcessing:
    """Text preprocessing module with advanced tokenization"""

    class Tokenizer:
        """Advanced tokenizer with MWE recognition and contraction handling"""

        def _init_(self, contractions_dict=None):
            self.contractions_dict = contractions_dict if contractions_dict is not None else {}
            try:
                self.wordnet = wn
                self.MWEs = self.list_MWEs() # converts new york to -> New_yrok - as its one entity
            except:
                print("WordNet initialization failed. Using basic tokenization.")
                self.MWEs = []
                #Replacing or extracting parts of strings.
                #Cleaning and preprocessing text.

            self.compile_regex_patterns()

        def list_MWEs(self):
            """Extract multi-word expressions (MWEs) from WordNet."""
            MWEs = []

            try:
                # Get multi-word nouns
                nouns = self.wordnet.synsets(pos="n")
                MWEs.extend([syn.lemmas()[0] for syn in nouns if " " in syn.lemmas()[0]])

                # Get multi-word verbs
                verbs = self.wordnet.synsets(pos="v")
                MWEs.extend([syn.lemmas()[0] for syn in verbs if " " in syn.lemmas()[0]])
            except:
                print("Error extracting MWEs. Using empty list.")

            return MWEs

        # predefiing rules
        def compile_regex_patterns(self):
            """Compile all required regex patterns."""
            # Multi-word expressions handling
            if self.MWEs:
                mwe_patterns = [rf"\b{re.escape(mwe)}\b" for mwe in self.MWEs]
                self.regex_pattern = re.compile("|".join(mwe_patterns))
            else:
                self.regex_pattern = re.compile(r"")

            # Hyphen handling (e.g., "high-quality" → "high quality")
            self.hyphen_pattern = re.compile(r"\b(\w+)-(\w+)\b")

            # Preserve numbers with units (e.g., "10kg" → "10_kg", "$100" → "$100")
            self.number_unit_pattern = re.compile(r"(\d+)([a-zA-Z]+)")

            # Punctuation removal (except in preserved cases)
            self.punctuation_pattern = re.compile(r"[^\w\s\-_]")

            # Contractions patterns
            self.contraction_pattern = re.compile(r"\b(" + "|".join(map(re.escape, self.contractions_dict.keys())) + r")\b", re.IGNORECASE)

        # follwo aboev rules
        def tokenize(self, text):
            """Tokenize the input text with preprocessing steps."""
            # Convert to lowercase
            text = text.lower()

            # Replace multi-word expressions with underscores
            if self.MWEs:
                text = self.regex_pattern.sub(lambda match: match.group(0).replace(" ", "_"), text)

            # Handle contractions
            text = self.contraction_pattern.sub(lambda match: self.contractions_dict.get(match.group(0).lower(), match.group(0)), text)

            # Handle hyphens (convert to spaces)
            text = self.hyphen_pattern.sub(r"\1 \2", text)

            # Preserve numbers with units
            text = self.number_unit_pattern.sub(r"\1_\2", text)

            # Remove other punctuation
            text = self.punctuation_pattern.sub("", text)

            # Tokenize by splitting on whitespace
            tokens = text.split()
            return tokens

   