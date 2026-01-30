FAKE NEWS DETECTION FOR STUDENTS USING MACHINE LEARNING 

------------------------------------------------------------
PROJECT OVERVIEW
------------------------------------------------------------
Misinformation spreads rapidly through online news platforms and social media,
making it difficult for students to differentiate between reliable and fake
information.

This project presents an AI-based Fake News Detection System designed specifically
for students. The system analyzes full-length news articles using Natural Language
Processing (NLP) and Machine Learning techniques to assess credibility, classify
news as Fake or Real, and generate concise summaries to help students quickly
understand the content before trusting or sharing it.

The project focuses on realistic, explainable, and ethical AI behavior rather than
forcing predictions on uncertain or insufficient input text.

The system also includes a Streamlit-based web interface for interactive student use

------------------------------------------------------------
PROBLEM STATEMENT
------------------------------------------------------------
Fake news spreads quickly through online news and social media, making it hard for
students to differentiate between reliable and fake information. There is a need
for an AI solution that can analyze articles, assess credibility, and provide
concise, trustworthy summaries to prevent the spread of false information.


------------------------------------------------------------
DATASET DETAILS
------------------------------------------------------------
Source:
Kaggle – Fake News Dataset

Dataset Structure Used:

Training Dataset:
dataset/archive (2)/training/training/fakeNewsDataset/
- fake/
- legit/

Testing Dataset:
dataset/archive (2)/Testing_dataset/testingSet/
- fake/
- real/

Label Mapping:
- fake  -> 0 (Fake News)
- legit / real -> 1 (Real News)

The dataset contains full-length news articles. Separate training and testing
folders are used to avoid data leakage and ensure fair evaluation.


------------------------------------------------------------
TECHNOLOGIES USED
------------------------------------------------------------
- Python
- Pandas
- Scikit-learn
- NLTK
- Matplotlib
- Seaborn
- Joblib
-Streamlit

------------------------------------------------------------
PROJECT STRUCTURE
------------------------------------------------------------
fake_news_detection_education/

|

|-- train_model.py        

|-- evaluate_model.py  

|-- visualize_data.py  

|-- predict_news.py 

|-- summarize_text.py 

|-- app.py

|

|-- fake_news_model.pkl 

|-- tfidf_vectorizer.pkl 

|-- requirements.txt

|-- README.txt


------------------------------------------------------------
MODEL BUILDING PIPELINE
------------------------------------------------------------
1. Data Cleaning:
   - Lowercasing text
   - Removing URLs, punctuation, numbers
   - Removing stopwords

2. Feature Extraction:
   - TF-IDF Vectorization

3. Model Used:
   - Logistic Regression

4. Evaluation Strategy:
   - Separate training and testing datasets
   - Classification report
   - Confusion matrix

5. Model Persistence:
   - Model and vectorizer saved using Joblib


------------------------------------------------------------
SUMMARIZATION MODULE
------------------------------------------------------------
The system includes an extractive text summarization feature.

Approach:
- Sentence tokenization
- Word frequency scoring
- Selection of most informative sentences

Purpose:
- Provide concise and trustworthy summaries
- Help students quickly understand verified content
- Avoid altering original meaning of the article


------------------------------------------------------------
VISUALIZATIONS (EDA)
------------------------------------------------------------
The following visualizations are generated:

1. Fake vs Real News Distribution
   - Shows dataset balance

2. Text Length Distribution
   - Compares word count of fake and real articles
   - Justifies rejection of very short inputs

3. Confusion Matrix
   - Displays true positives, false positives, true negatives, and false negatives


------------------------------------------------------------
MODEL PERFORMANCE
------------------------------------------------------------
- Accuracy on test dataset: ~65%
- Balanced precision, recall, and F1-score

The performance reflects the challenging nature of fake news detection in the
education domain, where fake and real news often share similar language.

The system prioritizes safety by flagging uncertain cases instead of forcing
confident predictions.


------------------------------------------------------------
HOW TO RUN THE PROJECT
------------------------------------------------------------

Step 1: Install dependencies
----------------------------
pip install -r requirements.txt


Step 2: Train the model
----------------------------------
python train_model.py

Creates:
- fake_news_model.pkl
- tfidf_vectorizer.pkl


Step 3: Evaluate the model
--------------------------
python evaluate_model.py

Outputs:
- Classification report
- Confusion matrix


Step 4: Visualize data
----------------------
python visualize_data.py

Outputs:
- Fake vs Real distribution graph
- Text length distribution graph


Step 5: Predict news and generate summary
-----------------------------------------
python predict_news.py

Instructions:
- Paste a full news paragraph (minimum 15 words)
- The system outputs:
  - Fake / Real prediction
  - Confidence scores
  - Concise summary


------------------------------------------------------------
IMPORTANT DESIGN DECISIONS
------------------------------------------------------------
- Short headlines are not classified to avoid unreliable predictions
- Probability thresholds are used to prevent false confidence
- Uncertain cases are flagged conservatively for student safety


------------------------------------------------------------
SOCIAL IMPACT
------------------------------------------------------------
- Helps students verify information before sharing
- Reduces spread of academic and educational misinformation
- Promotes digital literacy and critical thinking


------------------------------------------------------------
CONCLUSION
------------------------------------------------------------
This project demonstrates a complete and realistic machine learning workflow,
including data preprocessing, feature extraction, model training, evaluation,
prediction, and summarization.


------------------------------------------------------------
AUTHOR
------------------------------------------------------------
Aditya Seswani
