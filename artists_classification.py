#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
import argparse
import numpy as np
import pandas as pd
import random
from util import tokenized,get_metrics,train_predict_model,display_confusion_matrix,\
display_classification_report,display_model_performance_metrics
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB


def load_function_words(resource_path):
    """load a newline separated text file of function words.
    Return a list"""
    f_words = []
    with open(resource_path, 'r') as f:
        for line in f:
            if line.strip():
                f_words.append(line.lower().strip())
    return f_words


def load_features(list_of_lyrics, list_of_features):
    matrix = np.zeros((len(list_of_lyrics), len(list_of_features)), dtype=np.int)
    for i, lyric in enumerate(list_of_lyrics):
        lyric_words = lyric.split()
        for j, word in enumerate(list_of_features):
            these_words = [w for w in lyric_words if w == word]
            matrix[i,j] = len(these_words)
            
    return matrix


def main(data_file,vocab_path):
    original_data = pd.read_csv(data_file, encoding = 'utf-8')
    df = original_data[original_data['artist'].isin(['Donna Summer', 'Gordon Lightfoot','Bob Dylan','George Strait']) ]
    df['tokenized_text'] = tokenized(df['text'])
    
    random.seed(30)
    X = df['tokenized_text']
    Y = df['artist']
    # build train and test datasets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=100)
    
    ############ build TFIDF features on train text ##############
    tv = TfidfVectorizer(min_df=0.0, max_df=1.0, ngram_range=(1,2),sublinear_tf=True)
    tv_train_features = tv.fit_transform(X_train)
    tv_test_features = tv.transform(X_test)
    
    # Logistic Regression for TF-IDF
    lr = LogisticRegression(penalty='l2', max_iter=100, C=1)
    print('Logistic Regression for TF-IDF')
    lr_tfidf_predictions = train_predict_model(classifier=lr, 
                                             train_features=tv_train_features, train_labels=Y_train,
                                             test_features=tv_test_features, test_labels=Y_test)
    display_model_performance_metrics(true_labels=Y_test, predicted_labels=lr_tfidf_predictions,
                                      classes=['Donna Summer', 'Gordon Lightfoot','Bob Dylan','George Strait'])

    # build SVM for TF-IDF
    svm= SVC(kernel='linear')
    print('\nSVM for TF-IDF')
    SVM_tfidf_prediction=train_predict_model(classifier=svm, 
                                                train_features=tv_train_features, train_labels=Y_train,
                                                test_features=tv_test_features, test_labels=Y_test)
    display_model_performance_metrics(true_labels=Y_test, predicted_labels=SVM_tfidf_prediction,
                                      classes=['Donna Summer', 'Gordon Lightfoot','Bob Dylan','George Strait'])
    
    # build Random Forest for TF-IDF
    rfc = RandomForestClassifier(n_jobs=-1)
    print('\nRandom Forest for TF-IDF')
    rfc_tfidf_predictions = train_predict_model(classifier=rfc, 
                                                train_features=tv_train_features, train_labels=Y_train,
                                                test_features=tv_test_features, test_labels=Y_test)
    display_model_performance_metrics(true_labels=Y_test, predicted_labels=rfc_tfidf_predictions,
                                      classes=['Donna Summer', 'Gordon Lightfoot','Bob Dylan','George Strait'])
    
    # build Naive Bayes for TF-IDF
    NB = MultinomialNB()
    print('\nNaive Bayes for TF-IDF')
    NB_tfidf_prediction=train_predict_model(classifier=NB, 
                                                train_features=tv_train_features, train_labels=Y_train,
                                                test_features=tv_test_features, test_labels=Y_test)
    display_model_performance_metrics(true_labels=Y_test, predicted_labels=NB_tfidf_prediction,
                                      classes=['Donna Summer', 'Gordon Lightfoot','Bob Dylan','George Strait'])
    
    
    ################ build Bag of Words feature on train text ################
    cv = CountVectorizer(stop_words='english', max_features=10000,analyzer = 'word')
    cv_train_features = cv.fit_transform(X_train)
    cv_test_features = cv.transform(X_test)
    
    # build Logistic Regression for Bag of Words
    print('\nLogistic Regression for BoW')
    lr_bow_predictions = train_predict_model(classifier=lr, 
                                             train_features=cv_train_features, train_labels=Y_train,
                                             test_features=cv_test_features, test_labels=Y_test)
    display_model_performance_metrics(true_labels=Y_test, predicted_labels=lr_bow_predictions,
                                  classes=['Donna Summer', 'Gordon Lightfoot','Bob Dylan','George Strait'])
    
    # build SVM for BoW
    print('\nSVM for Bow')
    svm_bow_predictions = train_predict_model(classifier=svm, 
                                             train_features=cv_train_features, train_labels=Y_train,
                                             test_features=cv_test_features, test_labels=Y_test)
    display_model_performance_metrics(true_labels=Y_test, predicted_labels=svm_bow_predictions,
                                      classes=['Donna Summer', 'Gordon Lightfoot','Bob Dylan','George Strait'])
    
    # build Random Forest for Bow
    print('\nRandom Forest for Bow')
    rf_bow_predictions = train_predict_model(classifier=rfc, 
                                             train_features=cv_train_features, train_labels=Y_train,
                                             test_features=cv_test_features, test_labels=Y_test)
    display_model_performance_metrics(true_labels=Y_test, predicted_labels=rf_bow_predictions,
                                      classes=['Donna Summer', 'Gordon Lightfoot','Bob Dylan','George Strait'])
    
    # build Naive Bayes for Bow
    print('\nNaive Bayes for Bow')
    multiNB_bow_predictions = train_predict_model(classifier=NB, 
                                             train_features=cv_train_features, train_labels=Y_train,
                                             test_features=cv_test_features, test_labels=Y_test)
    display_model_performance_metrics(true_labels=Y_test, predicted_labels=multiNB_bow_predictions,
                                      classes=['Donna Summer', 'Gordon Lightfoot','Bob Dylan','George Strait'])
    

    ############## build function words feature on train text ############
    function_words = load_function_words(vocab_path)
    funcW_X_train = load_features(X_train,function_words)
    funcW_X_test = load_features(X_test,function_words)
    
    # build Logistic Regression for function words
    print('\nLogistic Regression for function words')
    lr_fw_predictions = train_predict_model(classifier=lr, 
                                             train_features=funcW_X_train, train_labels=Y_train,
                                             test_features=funcW_X_test, test_labels=Y_test)
    display_model_performance_metrics(true_labels=Y_test, predicted_labels=lr_fw_predictions,
                                  classes=['Donna Summer', 'Gordon Lightfoot','Bob Dylan','George Strait'])
    
    # build SVM for function words
    print('\nSVM for function words')
    svm_fw_predictions = train_predict_model(classifier=svm, 
                                             train_features=funcW_X_train, train_labels=Y_train,
                                             test_features=funcW_X_test, test_labels=Y_test)
    display_model_performance_metrics(true_labels=Y_test, predicted_labels=svm_fw_predictions,
                                      classes=['Donna Summer', 'Gordon Lightfoot','Bob Dylan','George Strait'])
    
    # build Random Forest for function words
    print('\nRandom Forest for function words')
    rf_fw_predictions = train_predict_model(classifier=rfc, 
                                             train_features=funcW_X_train, train_labels=Y_train,
                                             test_features=funcW_X_test, test_labels=Y_test)
    display_model_performance_metrics(true_labels=Y_test, predicted_labels=rf_fw_predictions,
                                      classes=['Donna Summer', 'Gordon Lightfoot','Bob Dylan','George Strait'])
    
    # build Naive Bayes for function words
    print('\nNaive Bayes for function words')
    multiNB_fw_predictions = train_predict_model(classifier=NB, 
                                             train_features=funcW_X_train, train_labels=Y_train,
                                             test_features=funcW_X_test, test_labels=Y_test)
    display_model_performance_metrics(true_labels=Y_test, predicted_labels=multiNB_fw_predictions,
                                      classes=['Donna Summer', 'Gordon Lightfoot','Bob Dylan','George Strait'])
    
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='song classification project')
    parser.add_argument('--path', type=str, default='songdata.csv',
                        help='path to author dataset')
    parser.add_argument('--function_words_path', type=str, default="ewl_function_words.txt",
                       help='path to the list of words to use as features')
    args = parser.parse_args()
    #main(args.path)

    main(args.path, args.function_words_path)