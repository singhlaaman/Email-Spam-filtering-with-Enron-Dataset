#This is code to get ROC curve for SVM.
import os
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp


def make_Dictionary(root_dir):
    emails_dirs = [os.path.join(root_dir,f) for f in os.listdir(root_dir)]    
    all_words = []       
    for emails_dir in emails_dirs:
        dirs = [os.path.join(emails_dir,f) for f in os.listdir(emails_dir)]
        for d in dirs:
            emails = [os.path.join(d,f) for f in os.listdir(d)]
            for mail in emails:
                with open(mail) as m:
                    for line in m:
                        words = line.split()
                        all_words += words
    dictionary = Counter(all_words)
    list_to_remove = dictionary.keys()
    
    for item in list_to_remove:
        if item.isalpha() == False: 
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]
    dictionary = dictionary.most_common(3000)
    
    np.save('dict_enron.npy',dictionary) 
    
    return dictionary
    
def extract_features(root_dir): 
    emails_dirs = [os.path.join(root_dir,f) for f in os.listdir(root_dir)]  
    docID = 0
    features_matrix = np.zeros((33716,3000))
    train_labels = np.zeros(33716)
    for emails_dir in emails_dirs:
        dirs = [os.path.join(emails_dir,f) for f in os.listdir(emails_dir)]
        for d in dirs:
            emails = [os.path.join(d,f) for f in os.listdir(d)]
            for mail in emails:
                with open(mail) as m:
                    all_words = []
                    for line in m:
                        words = line.split()
                        all_words += words
                    for word in all_words:
                      wordID = 0
                      for i,d in enumerate(dictionary):
                        if d[0] == word:
                          wordID = i
                          features_matrix[docID,wordID] = all_words.count(word)
                train_labels[docID] = int(mail.split(".")[-2] == 'spam')
                docID = docID + 1                
    return features_matrix,train_labels
    
#Create a dictionary of words with its frequency

root_dir = 'Enron-data-set'
dictionary = make_Dictionary(root_dir)


#Prepare feature vectors per training mail and its labels

features_matrix,labels = extract_features(root_dir)
np.save('enron_features_matrix.npy',features_matrix)
np.save('enron_labels.npy',labels)


#train_matrix = np.load('enron_features_matrix.npy');
#labels = np.load('enron_labels.npy');
print(features_matrix.shape)
print(labels.shape)
print(sum(labels==0),sum(labels==1))
n_classes = labels.shape[1]
X_train, X_test, y_train, y_test = train_test_split(features_matrix, labels, test_size=0.40)

# Learn to predict each class against the other
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=random_state))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()