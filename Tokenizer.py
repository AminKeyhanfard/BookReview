import enum
import re
from collections import Counter
import nltk
from nltk import PorterStemmer
from nltk.corpus import stopwords

class TokenOptions(enum.Enum):
    REMOVE_PUNCTUATION = 1
    KEEP_PUNCTUATION = 2
    REMOVE_STOPWORDS = 3
    KEEP_STOPWORDS = 4
    APPLY_STEMMING = 5
    NO_STEMMING = 6

class DATokenizer:
    def __init__(self, punctuation_opt: TokenOptions, stopword_opt: TokenOptions, stem_opt: TokenOptions):
        self.punctuation_opt = punctuation_opt
        self.stopword_opt = stopword_opt
        self.stem_opt = stem_opt
        # nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))
        
        ## to discover the word clouds without these words
        self.stop_words.add('book')
        self.stop_words.add('books')
        
        self.stemmer = PorterStemmer()

    def tokenize(self, text: str) -> list:
        # change everything to lower case
        text = text.lower()

        # Processing punctuations
        if self.punctuation_opt == TokenOptions.REMOVE_PUNCTUATION:
            text = re.sub(r'[^\w\s]', '', text)
        tokens = text.split()

        # Processing stopwords
        if self.stopword_opt == TokenOptions.REMOVE_STOPWORDS:
            tokens = [t for t in tokens if t not in self.stop_words]

        # Handling Stemming
        if self.stem_opt == TokenOptions.APPLY_STEMMING:
            tokens = [self.stemmer.stem(t) for t in tokens]
        return tokens

    def getSortedWordFrequencies(self, token_list: list):
        token_counter = Counter(token_list)
        final_dict = dict(
            sorted(token_counter.items(), key=lambda item: item[1], reverse=True))  # Descending by value
        return final_dict