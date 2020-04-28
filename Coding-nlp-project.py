#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#libraries and functions used in this project
import pandas as pd 
import numpy as np
import nltk
nltk.download('punkt')
import re
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import gensim
from collections import Counter
import math
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from simpletransformers.classification import ClassificationModel
import logging

def confusion_matrix(y_val,predicted,d):
    conf_mat= np.zeros((d,d))
    for i in range(len(y_val)):
        conf_mat[list(y_val)[i]-1][predicted[i]-1] += 1
    return conf_mat

def remove_punct(text):
    text_nopunct = ''
    text_nopunct = re.sub('['+string.punctuation+']', '', text)
    return text_nopunct

def accuracy(predicted,y_val):
    accuracy= sum(predicted==list(y_val))/len(y_val)
    return accuracy
       
def randomforest(matrix,labels):
    X_train, X_val, y_train, y_val = train_test_split(matrix, labels, test_size=0.2, random_state=0)
    randomforest = RandomForestClassifier(max_depth=None, random_state=0, n_estimators=20).fit(X_train, y_train)
    predicted = randomforest.predict(X_val)
    return X_train, X_val, y_train, y_val, predicted

def logistic_regression(matrix,labels):
    X_train, X_val, y_train, y_val = train_test_split(matrix, labels, test_size=0.2, random_state=0)
    logreg = LogisticRegression().fit(X_train, y_train)
    predicted = logreg.predict(X_val)
    return X_train, X_val, y_train, y_val, predicted

def pre_rec_F1(cm_all):
    labels = [1,2,3,4,5]
    #general level evaluations
    cm_norm= cm_all.astype('float') / cm_all.sum(axis=1)[:, np.newaxis]
    cm_norm2= cm_all.astype('float') / cm_all.sum(axis=0)
    print("Recall and Precision:")
    F_all = []
    for c in range(len(labels)):
        p = cm_norm2[c,c]
        r = cm_norm[c,c]
        F = 2*p*r/(p+r)
        F_all.append(F)
        print("R: %.2f" % (r)+",\t P: %.2f" %(p)+",\t F1: %.2f" %(F)+",\t for n="+              str(int(sum(cm_all[c,:])))+"\t "+str(labels[c]))
    print("\nAverage recall: %.2f " %(np.mean(np.diag(cm_norm))))
    print("Average precision: %.2f " %(np.mean(np.diag(cm_norm2)))) 
    print("Average F1 score: %.2f " %(np.mean(F_all)))
    

def clean_data(review,stop_words):
    reviews_no_punct = review.apply(lambda x: remove_punct(x))
    tokenized_reviews = []
    for i in reviews_no_punct:
        tokens=nltk.word_tokenize(i.lower())
        tokens_clean = []
        for token in tokens:
            if not token in stop_words:
                tokens_clean.append(token)
        tokenized_reviews.append(tokens_clean)
    return tokenized_reviews

def flat_list(list):
    all_words = []
    for sublist in list:
        for item in sublist:
            all_words.append(item)
    return all_words

def review_count_matrix(tokenized_reviews,word_list):
    num_col = len(tokenized_reviews)
    num_row = len(word_list)
    matrix = np.zeros((num_row,num_col))
    count_dict = []
    for rev in tokenized_reviews:
        count_dict.append(Counter(rev))
    for col in range(num_col):
        for row in range(num_row):
            matrix[row][col] = count_dict[col][word_list[row]]
    return matrix


# In[ ]:


#downloading the Google News data set for the the second method
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.gz', binary=True)


# In[ ]:


#Reading data and two columns of it as reviews and labels 
data = pd.read_table("amazon_alexa.tsv")
reviews = data.verified_reviews
labels = data.rating
d = 5 

#Preprocessing and Cleaning the data
#removing stop words and punctuation marks, lowercasing and tokenizing :
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
tokenized_reviews = clean_data(reviews,stop_words)

#flatting the list of tokenized words to have all the word in one list and then unique them
all_words = flat_list(tokenized_reviews)       
all_words_uniq = list(set(all_words))


# In[ ]:


len(all_words_uniq)


# In[ ]:


#plotting top 50 words appeared in reviews 
frequency_dict = nltk.FreqDist(all_words)
print(sorted(frequency_dict,key=frequency_dict.__getitem__, reverse=True)[0:50])

wordcloud = WordCloud()
wordcloud.generate_from_frequencies(frequency_dict)

plt.figure(figsize=(10,10))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# In[ ]:


#First method: bag of words method
#building word-document matrix
doc_word_matrix = np.zeros((len(tokenized_reviews), len(all_words_uniq)))
vocab_dict = {all_words_uniq[i]:i for i in range(len(all_words_uniq))}
for row in range(len(tokenized_reviews)):
    for word in tokenized_reviews[row]:
        doc_word_matrix[row][vocab_dict[word]] += 1
        
#Randomforest classification: training and predicting 
X_train_ranf, X_val_ranf, y_train_ranf, y_val_ranf, predicted_randf = randomforest(doc_word_matrix,labels)
#building confusion matrix
conf_mat_randf = confusion_matrix(y_val_ranf,predicted_randf,d)
#printing results
accuracy_randf = accuracy(predicted_randf,y_val_ranf)
print("Random Forest results for bag-of-words method: \n")
print("validation accuaracy:",accuracy_randf)
print("\n confusion matrix \n", conf_mat_randf)
pre_rec_F1(conf_mat_randf)

#Logistic Regression classification: training and predicting
X_train_logr, X_val_logr, y_train_logr, y_val_logr, predicted_logr = logistic_regression(doc_word_matrix,labels)
#building confusion matrix
conf_mat_logr = confusion_matrix(y_val_logr,predicted_logr,d)
#printing results
accuracy_logr= accuracy(predicted_logr,y_val_logr)
print("\n Logistic Regression results for bag-of-words method: \n")
print("validation accuaracy:",accuracy_logr)
print("\n confusion matrix \n", conf_mat_logr)
pre_rec_F1(conf_mat_logr)


# In[ ]:


#Second mathod: weighted word2vec
#getting words and word2vec of these words
wrod2vec_matrix =[]
word_list = []
for word in all_words_uniq:
    if word not in model.vocab:
        pass
    if word in model.vocab:
        word_list.append(word)
        wrod2vec_matrix.append(model[word])
#Building tf-idf document-word2vec matrix
word2vec_vocab_dict = {word_list[i]:i for i in range(len(word_list))}
doc_word2vec_matrix = np.zeros((len(tokenized_reviews),300))
matrix = review_count_matrix(tokenized_reviews,word_list)
for rev in range(len(tokenized_reviews)):
    doc = []
    count_dict = Counter(tokenized_reviews[rev])
    for i in range(len(tokenized_reviews[rev])):
        if tokenized_reviews[rev][i] in word_list:
            tf = count_dict[tokenized_reviews[rev][i]]
            counter = 0
            for k in range(len(tokenized_reviews)):
                if tokenized_reviews[rev][i] in tokenized_reviews[k]:
                    counter += 1
            idf = math.log10((len(tokenized_reviews))/counter)
            tf_idf_score = tf*idf
            doc.append(wrod2vec_matrix[word2vec_vocab_dict[tokenized_reviews[rev][i]]]*tf_idf_score)
    if len(doc)!=0:
        avg_vec_doc = np.average(doc, axis=0)  
        doc_word2vec_matrix[rev,:] = avg_vec_doc

        
#Logistic Regression classification: training and predection 
X_train_w2vlr, X_val_w2vlr, y_train_w2vlr, y_val_w2vlr, predicted_w2vlr = logistic_regression(doc_word2vec_matrix, labels)
#building confusion matrix
conf_mat_w2vlr = confusion_matrix(y_val_w2vlr,predicted_w2vlr,d)
#printing results
accuracy_w2vlr= accuracy(predicted_w2vlr,y_val_w2vlr)
print("\n Logistic Regression results for word2vec method: \n")
print("validation accuaracy:",accuracy_w2vlr)
print("\n confusion matrix \n", conf_mat_w2vlr)
pre_rec_F1(conf_mat_w2vlr)

#Randomforest classification: training and predicting 
X_train_w2vrf, X_val_w2vrf, y_train_w2vrf, y_val_w2vrf, predicted_w2vrf = randomforest(doc_word2vec_matrix,labels)
#building confusion matrix
conf_mat_w2vrf = confusion_matrix(y_val_w2vrf,predicted_w2vrf,d)
#printing results
accuracy_w2vrf= accuracy(predicted_w2vrf,y_val_w2vrf)
print("\n Random FOrest results for word2vec method: \n")
print("validation accuaracy:",accuracy_w2vrf)
print("\n confusion matrix \n", conf_mat_w2vrf)
pre_rec_F1(conf_mat_w2vrf)


# In[ ]:


#Third method: transformer 
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

#we need to read the data differently. so we will have list of lists, where each list contains one review and its
# label in the data.
reviews = data.verified_reviews
labels = data.rating -1
new_data = []
for i in range(len(reviews)):
    doc = []
    doc.append(reviews[i])
    doc.append(labels[i])
    new_data.append(doc)

#Splitting data to training and validation sets    
train,val = train_test_split(new_data, test_size=0.2, random_state=0)
training_df = pd.DataFrame(train)
validation_df = pd.DataFrame(val)

#downloading the model
model = ClassificationModel('bert', 'bert-base-cased', num_labels=5, 
                            args={'reprocess_input_data': True, 'overwrite_output_dir': True,"num_train_epochs": 4}
                            ,use_cuda=False) 
#training and prediction
model.train_model(training_df)
predictions, raw_outputs = model.predict(list(validation_df[0]))
y_val = list(validation_df[1])

#building confusion matrix
conf_mat_trans = confusion_matrix(y_val,predictions,d)
#printing results
accuracy_trans= accuracy(predictions,y_val)
print("\n Logistic Regression results for word2vec method: \n")
print("validation accuaracy:",accuracy_trans)
print("\n confusion matrix \n", conf_mat_trans)
pre_rec_F1(conf_mat_trans)

