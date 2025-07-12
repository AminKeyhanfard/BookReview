
# 📚 Book Review Data Analysis Project

This project analyzes book metadata and reviews to gain insights through data cleaning, visualization, sentiment analysis, and predictive modeling. It is split into two main components:

---

## 📁 Part 1: Book Metadata & Review Cleaning

### 🔹 Features:
- Loads datasets: `books_data.csv` and `books_ratings.csv`
- Cleans and processes fields like `title`, `authors`, `description`, and `ratingsCount`
- Saves cleaned outputs as `clean_books.csv` and `clean_ratings.csv`

⚠️ `books_data.csv` and `books_ratings.csv` in the repository are limited to 10,000 rows due to size restrictions.

### 🔹 Output Files:
- `clean_books.csv`
- `clean_ratings.csv`

### 📦 Required Libraries:
```bash
pandas, matplotlib, wordcloud
```

---

## 🧠 Part 2: Sentiment Analysis and Rating Prediction

### 🔹 Features:
- Converts `Score` to polarity sentiment: Positive (4–5), Neutral (3), Negative (1–2)
- Cleans review text (removes stopwords, punctuation)
- Generates frequency-based word clouds for positive/negative/neutral reviews
- Trains and evaluates a logistic regression classifier for:
    - Predicting sentiment (`+1`, `-1`)
    - Predicting actual review scores (1–5)

### 🔹 Model:
- TF-IDF Vectorization
- Logistic Regression (from scikit-learn)
- Random sampling for balanced classification

### 🔹 Outputs:
- Word clouds by polarity (summary and text)
- Rating distribution histogram
- Accuracy scores for:
    - Sentiment classification
    - Rating prediction

### 📦 Required Libraries:
```bash
pandas, numpy, matplotlib, wordcloud, sklearn
Tokenizer.py (custom tokenizer module)
```

### ▶️ How to Run:
```bash
python cleaning_script.py            # For Part 1
python sentiment_analysis.py         # For Part 2
```

---

## 📌 Folder Structure:
```
.
├── books_data.csv
├── books_ratings.csv
├── clean_books.csv
├── clean_ratings.csv
├── cleaning_script.py
├── sentiment_analysis.py
├── Tokenizer.py
├── README.md / README.docx
```

---

## 🙌 Author
**Amin Keyhanfard**  
GitHub: [@AminKeyhanfard](https://github.com/AminKeyhanfard)
