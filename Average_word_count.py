# This is code to find average word size in Ham emails and Spam Emails
import os

def word_count(path):
    all_words = []
    emails = [os.path.join(path,f) for f in os.listdir(path)]
    for mail in emails:
        with open(mail, encoding="latin1") as m:
            for line in m:
                words = line.split()
                all_words += words
    x = len(all_words)
    y = len(os.listdir(path))
    avg_wordcount = x/y
    return avg_wordcount

ham_path = 'C:/Users/sanee/Train_Set_Split/ham/'
spam_path = 'C:/Users/sanee/Train_Set_Split/spam/'

ham_wordCount = word_count(ham_path)
print('Average Word Count of Ham mails:',ham_wordCount)

spam_wordCount = word_count(spam_path)
print('Average Word Count of Spam mails:',spam_wordCount)