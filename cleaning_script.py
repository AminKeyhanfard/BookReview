import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud


class BookReviewProject:
    def run_project(self):
        """Main method to run all steps of the project."""
        self.load_data()
        self.clean_data()
        self.save_clean_data()

    def load_data(self):
        """Loads the book metadata and rating datasets from CSV files."""
        print('\nLoading CSV files...')
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)

        self.books_data = pd.read_csv("books_data.csv")
        self.books_ratings = pd.read_csv("books_ratings.csv")

        print("\nBooks Data Info:")
        self.books_data.info()

        print("\nBooks Ratings Info:")
        self.books_ratings.info()

    def clean_data(self):
        """Cleans and preprocesses both datasets."""
        print('\nCleaning data...')

        # Clean books metadata
        clean_books = self.books_data.copy()
        clean_books.columns = [
            'Title', 'Description', 'Authors', 'Image', 'PreviewLink',
            'Publisher', 'PublishedDate', 'InfoLink', 'Categories',
            'RatingsCount'
        ]

        clean_books = clean_books[['Title', 'Description', 'Authors', 'Publisher',
                                   'PublishedDate', 'Categories', 'RatingsCount']]

        print('\nRows before cleaning (books):', len(clean_books))
        clean_books.dropna(subset=['Title'], inplace=True)

        clean_books['Title'] = clean_books['Title'].astype(str)
        clean_books['Description'] = clean_books['Description'].fillna('Unknown').astype(str)
        clean_books['Authors'] = clean_books['Authors'].fillna('Unknown').astype(str)
        clean_books['Publisher'] = clean_books['Publisher'].fillna('Unknown').astype(str)
        clean_books['RatingsCount'] = clean_books['RatingsCount'].fillna(0).astype(int)
        clean_books['PublishedDate'] = pd.to_datetime(clean_books['PublishedDate'], errors='coerce')
        clean_books['Categories'] = clean_books['Categories'].fillna('Unknown').astype(str)

        print('Rows after cleaning (books):', len(clean_books))
        self.clean_books = clean_books

        # Clean book ratings data
        clean_ratings = self.books_ratings.copy()
        clean_ratings.columns = [
            'Id', 'Title', 'Price', 'UserId', 'ProfileName',
            'Helpfulness', 'Score', 'Time', 'Summary', 'Text'
        ]

        clean_ratings = clean_ratings[['Title', 'Score', 'Summary', 'Text']]

        print('\nRows before cleaning (ratings):', len(clean_ratings))
        clean_ratings.dropna(subset=['Title'], inplace=True)

        clean_ratings['Title'] = clean_ratings['Title'].astype(str)
        clean_ratings['Score'] = clean_ratings['Score'].fillna(0).astype(int)
        clean_ratings['Summary'] = clean_ratings['Summary'].fillna('').astype(str)
        clean_ratings['Text'] = clean_ratings['Text'].fillna('').astype(str)

        print('Rows after cleaning (ratings):', len(clean_ratings))
        self.clean_ratings = clean_ratings

    def save_clean_data(self):
        """Saves the cleaned datasets to CSV files."""
        print('\nSaving cleaned datasets to CSV...')
        self.clean_books.to_csv('clean_books.csv', index=False)
        self.clean_ratings.to_csv('clean_ratings.csv', index=False)


# Run the project only when this script is executed directly
if __name__ == "__main__":
    project = BookReviewProject()
    project.run_project()
