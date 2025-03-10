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
    

    class tf_idf_Vectorizer:
        """TF-IDF Vectorizer implementation with advanced features"""

        def _init_(self, max_features=None):
            self.vocabulary = {}
            self.idf = {}
            self.max_features = max_features
            self.fitted = False

        def fit(self, corpus):
            """Build vocabulary with unique indices and compute IDF values."""
            if isinstance(corpus[0], str):
                # If corpus is a list of strings, tokenize them
                tokenizer = PreProcessing.Tokenizer()
                corpus = [tokenizer.tokenize(doc) for doc in corpus]
            elif not isinstance(corpus[0], list):
                raise ValueError("Corpus must be a list of strings or tokenized documents (list of lists).")

            # Count document frequency (DF) for each word
            df = Counter()
            for doc in corpus:
                unique_words = set(doc)
                df.update(unique_words)

            # Sort words by DF in descending order
            sorted_words = [word for word, _ in df.most_common(self.max_features)] if self.max_features else list(df.keys())

            # Create vocabulary
            self.vocabulary = {word: idx for idx, word in enumerate(sorted_words)}

            # Compute IDF with smoothing: log((N + 1) / (df + 1)) + 1
            N = len(corpus)
            self.idf = {word: np.log((N + 1) / (df[word] + 1)) + 1 for word in self.vocabulary}

            self.fitted = True
            return self

        def transform(self, documents):
            """Convert new documents into TF-IDF vectors using learned vocabulary."""
            if not self.fitted:
                raise ValueError("Vectorizer needs to be fitted before transform")

            if isinstance(documents[0], str):
                # If documents is a list of strings, tokenize them
                tokenizer = PreProcessing.Tokenizer()
                documents = [tokenizer.tokenize(doc) for doc in documents]
            elif not isinstance(documents[0], list):
                raise ValueError("Input documents must be strings or tokenized documents (list of lists).")

            tfidf_matrix = np.zeros((len(documents), len(self.vocabulary)))

            for i, doc in enumerate(documents):
                # Term frequency for the document
                tf = Counter(doc)
                total_words = len(doc)

                for word, count in tf.items():
                    if word in self.vocabulary:  # Ignore unseen words
                        word_idx = self.vocabulary[word]
                        tfidf_matrix[i][word_idx] = (count / total_words) * self.idf.get(word, 0)

            return tfidf_matrix

        def fit_transform(self, documents):
            """Fit and transform documents"""
            self.fit(documents)
            return self.transform(documents)       

   