import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import Tokenizer as dat  # Ensure Tokenizer.py is in the same directory

class BookReviewProject:
    def runProject(self):
        # Step 1: Load the cleaned review data
        self.loadData()

        # Step 2: Analyze sentiments and visualize them
        self.analyzeSentiments()

        # Step 3: Predict polarity (positive/negative sentiment)
        self.predictSentiment()

        # Step 4: Predict review ratings
        self.predictRating()

    def loadData(self):
        print("\nLoading cleaned CSV with review data...")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)

        pathBooksRating = "clean_ratings.csv"
        self.books_rating = pd.read_csv(pathBooksRating)

        # Limit the size for faster computation and memory efficiency
        self.books_rating = self.books_rating[0:10000]
        print("Data Loaded Successfully.")

    def analyzeSentiments(self):
        print("\nAnalyzing Sentiments...")

        # Assign polarity: 1 (positive), -1 (negative), 0 (neutral)
        self.books_rating['polarity'] = [-1 if r in [1, 2]
                                         else 0 if r == 3
                                         else 1
                                         for r in self.books_rating['Score']]

        # Ensure text fields are strings
        self.books_rating['Summary'] = self.books_rating['Summary'].astype(str)
        self.books_rating['Text'] = self.books_rating['Text'].astype(str)

        # Separate positive and negative reviews
        self.pos_df = self.books_rating[self.books_rating.polarity == 1]
        self.neg_df = self.books_rating[self.books_rating.polarity == -1]

        # Plot histogram of rating scores
        plt.figure(figsize=(15, 12))
        plt.hist(self.books_rating['Score'], bins=5, color=(1, 0.8, 0.7),
                 alpha=0.7, edgecolor='black')
        plt.title("Rating Distribution", fontweight='bold', fontsize=30)
        plt.xlabel("Review Score", fontweight='bold', fontsize=18)
        plt.ylabel("Frequency", fontweight='bold', fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=18)
        plt.show()

        # Initialize tokenizer
        tokenizer = dat.DATokenizer(dat.TokenOptions.REMOVE_PUNCTUATION,
                                    dat.TokenOptions.REMOVE_STOPWORDS,
                                    dat.TokenOptions.NO_STEMMING)

        # Tokenize and get word frequencies
        all_text_freq = tokenizer.getSortedWordFrequencies(
            tokenizer.tokenize(' '.join(self.books_rating['Text'])))
        pos_text_freq = tokenizer.getSortedWordFrequencies(
            tokenizer.tokenize(' '.join(self.pos_df['Text'])))
        neg_text_freq = tokenizer.getSortedWordFrequencies(
            tokenizer.tokenize(' '.join(self.neg_df['Text'])))
        all_title_freq = tokenizer.getSortedWordFrequencies(
            tokenizer.tokenize(' '.join(self.books_rating['Summary'])))
        pos_title_freq = tokenizer.getSortedWordFrequencies(
            tokenizer.tokenize(' '.join(self.pos_df['Summary'])))
        neg_title_freq = tokenizer.getSortedWordFrequencies(
            tokenizer.tokenize(' '.join(self.neg_df['Summary'])))

        # Generate word clouds
        print("\nGenerating Word Clouds...")
        wc_dict = {
            "ALL TEXT": WordCloud(width=1600, height=1200, background_color='white').generate_from_frequencies(all_text_freq),
            "POS TEXT": WordCloud(width=1600, height=1200, background_color='white').generate_from_frequencies(pos_text_freq),
            "NEG TEXT": WordCloud(width=1600, height=1200, background_color='white').generate_from_frequencies(neg_text_freq),
            "ALL SUMMARY": WordCloud(width=1600, height=1200, background_color='white').generate_from_frequencies(all_title_freq),
            "POS SUMMARY": WordCloud(width=1600, height=1200, background_color='white').generate_from_frequencies(pos_title_freq),
            "NEG SUMMARY": WordCloud(width=1600, height=1200, background_color='white').generate_from_frequencies(neg_title_freq)
        }

        # Plotting word clouds
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        titles = list(wc_dict.keys())
        clouds = list(wc_dict.values())
        idx = 0
        for i in range(2):
            for j in range(3):
                axes[i, j].imshow(clouds[idx], interpolation='bilinear')
                axes[i, j].axis('off')
                axes[i, j].set_title(titles[idx])
                idx += 1

        plt.tight_layout()
        plt.show()

    def remove_punc_stopwords(self, text):
        tokenizer = dat.DATokenizer(dat.TokenOptions.REMOVE_PUNCTUATION,
                                    dat.TokenOptions.REMOVE_STOPWORDS,
                                    dat.TokenOptions.NO_STEMMING)
        return ' '.join(tokenizer.tokenize(text))

    def predictSentiment(self):
        print("\nPredicting Sentiment Polarity...")

        # Clean text
        df = self.books_rating.copy()
        df['Text'] = df['Text'].apply(self.remove_punc_stopwords)

        # Balance the dataset
        pos_df = df[df.polarity == 1]
        neg_df = df[df.polarity == -1]
        size = min(len(pos_df), len(neg_df))
        df = pd.concat([pos_df.sample(size), neg_df.sample(size)])

        # Split train/test using a random index
        df['rand'] = np.random.rand(len(df))
        train_df = df[df['rand'] < 0.85]
        test_df = df[df['rand'] >= 0.85]

        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(token_pattern=r'\b\w+\b')
        X_train = vectorizer.fit_transform(train_df['Text'])
        X_test = vectorizer.transform(test_df['Text'])

        y_train = train_df['polarity']
        y_test = test_df['polarity']

        # Train and evaluate Logistic Regression
        model = LogisticRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Accuracy
        accuracy = np.mean(predictions == y_test)
        print(f"Sentiment Prediction Accuracy: {accuracy:.2%}")

    def predictRating(self):
        print("\nPredicting Review Rating...")

        # Clean text
        df = self.books_rating.copy()
        df['Text'] = df['Text'].apply(self.remove_punc_stopwords)

        # Train/test split
        train_df, test_df = train_test_split(df, test_size=0.15)

        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(token_pattern=r'\b\w+\b')
        X_train = vectorizer.fit_transform(train_df['Text'])
        X_test = vectorizer.transform(test_df['Text'])

        y_train = train_df['Score']
        y_test = test_df['Score']

        # Logistic Regression
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Accuracy
        accuracy = np.mean(predictions == y_test)
        print(f"Rating Prediction Accuracy: {accuracy:.2%}")

# Run the project
if __name__ == "__main__":
    project = BookReviewProject()
    project.runProject()