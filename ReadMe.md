## Overview
Our topic is about 'Predicting Artists by the Lyrics'. Authorship detection is an essential part of the Natural Language Process, it helps us to identify the authors of texts. In this project, authorship detection was applied to predict the artists of lyrics. We dealt with this problem by implementing different classification machine learning algorithms to different features that were vectorized by TF-IDF, Bag of words and distribution of function words in each song lyrics where the function words are from the essential word list of 176.  Among all these models we utilized, the Support Vector Machine (SVM) classifier on TF-IDF gained the best accuracy of 79%, and Logistic Regression achieved the highest accuracy of 68.4% on bag of words and Naive Bayes performed the highest accuracy of 46% on distribution of function words.
## Files

### `songdata.csv`
The data we applied in this project. It is from Kaggle https://www.kaggle.com/mousehead/songlyrics

### `ewl_function_words.txt` 
It is the essential words list of 176 function words

### `util.py` 
Contains functions of tokenization and evaluations

### `artists_classification.py` 
Include vecorization and model function and it also call functions from util.py

Usage: `python artists_classification.py --function_words_path ewl_function_words.txt --path songdata.csv`
