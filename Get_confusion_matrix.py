#This is code to get the confusion matrix
import os
import numpy as np
import nltk

from nltk.corpus import stopwords
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC

def make_Dictionary(path):
    emails = [os.path.join(path,f) for f in os.listdir(path)]    
    all_words = []       
    for email in emails:
        with open(email, encoding='latin1') as m:
            content = m.read()
            all_words += nltk.word_tokenize(content)
    dictionary = [word for word in all_words if word not in stopwords.words('english')]
    dictionary = [word.lower() for word in dictionary if word.isalpha()]
    dictionary = Counter(dictionary)
    dictionary = dictionary.most_common(3000)
    return dictionary

def extract_features_train(path): 
    docID = 0
    features_matrix = np.zeros((23592,3000))
    labels = np.zeros(23592)
    emails = [os.path.join(path,f) for f in os.listdir(path)]
    for mail in emails:
        with open(mail, encoding="latin1") as m:
            all_words = []
            for line in m:
                words = line.split()
                all_words += words
            for word in all_words:
                wordID = 0
                for i,d in enumerate(dicti):
                    if d[0] == word:
                        wordID = i
                        features_matrix[docID,wordID] = all_words.count(word)
        labels[docID] = int(mail.split(".")[-2] == 'spam')
        docID = docID + 1                
    return features_matrix,labels

def extract_features_test(test_dir): 
    docID = 0
    features_matrix = np.zeros((10111,3000))
    labels = np.zeros(10111)
    emails = [os.path.join(test_dir,f) for f in os.listdir(test_dir)]
    for mail in emails:
        with open(mail, encoding="latin1") as m:
            all_words = []
            for line in m:
                words = line.split()
                all_words += words
            for word in all_words:
                wordID = 0
                for i,d in enumerate(dicti):
                    if d[0] == word:
                        wordID = i
                        features_matrix[docID,wordID] = all_words.count(word)
        labels[docID] = int(mail.split(".")[-2] == 'spam')
        docID = docID + 1                
    return features_matrix,labels

path = '/Users/vikaschhillar/Downloads/machine data/trainenron/'
dicti = make_Dictionary(path)

train_matrix,train_labels = extract_features_train(path)

model1 = MultinomialNB()
model2 = LinearSVC()
model1.fit(train_matrix,train_labels)
model2.fit(train_matrix,train_labels)

test_dir = '/Users/vikaschhillar/Downloads/machine data/testenron/'
test_matrix,test_labels = extract_features_test(test_dir)

result1 = model1.predict(test_matrix)
result2 = model2.predict(test_matrix)
print(confusion_matrix(test_labels,result1))
print(confusion_matrix(test_labels,result2))