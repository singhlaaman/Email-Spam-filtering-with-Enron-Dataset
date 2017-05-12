# This is code to get top 10 words for emails of Ham and Spam.
import os
from shutil import copyfile
import random
import nltk
import os
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

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
    dictionary = dictionary.most_common(10)
    return dictionary

spam_path = '/Users/vikaschhillar/Downloads/machine data/ham'

spam_dict = make_Dictionary(spam_path)

x = []
y = []
my_xticks = []
width = 1/1.5

for i in range(len(spam_dict)):
    x.append(i)
    y.append(spam_dict[i][1])
    my_xticks.append(spam_dict[i][0])

plt.xticks(x, my_xticks)
plt.bar(x, y, width, color="red")
plt.show()