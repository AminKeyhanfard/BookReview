Book Review Data Analysis Project
📁 Project Structure
.
├── book_review_project.py     # Main script
├── books_data.csv             # Raw book metadata
├── Books_rating.csv           # Raw book ratings and reviews
├── clean_books.csv            # Cleaned metadata (generated)
├── clean_ratings.csv          # Cleaned ratings data (generated)
└── README.md                  # Project documentation
📌 Features
- 📖 Data Cleaning  
  Handles missing values, renames columns, standardizes data types, and filters relevant fields.
  
- 📊 Visualizations  
  - Histograms of rating counts (including log-scale)
  - Word cloud from book descriptions

- 💾 Data Export  
  Saves cleaned datasets to `clean_books.csv` and `clean_ratings.csv`.
🔧 Requirements
Install the required libraries with:

```
pip install pandas matplotlib wordcloud
```
🚀 How to Run
1. Ensure `books_data.csv` and `Books_rating.csv` are in the same directory.
2. Run the script:

```
python book_review_project.py
```

3. View terminal output for data summaries, and see the visualizations displayed in new windows.
📈 Example Outputs
- Histogram of Ratings Count
- Log-Scaled Histogram
- Word Cloud of Book Descriptions
📚 Data Sources
This project uses two datasets:
- `books_data.csv` – Book metadata (title, authors, publisher, categories, ratings count)
- `Books_rating.csv` – User-generated ratings and reviews

*Note: Ensure data is properly licensed and anonymized if sourced externally.*
💡 Future Improvements
- Add sentiment analysis for user reviews  
- Combine metadata with rating scores for deeper insights  
- Build a simple recommendation engine
🧑‍💻 Author
Developed by Amin Keyhanfard  
📌 GitHub: https://github.com/AminKeyhanfard

Feel free to explore more projects and reach out with questions or collaboration ideas!
