#!/usr/bin/python

import pickle
import sys
import re
import os.path
import os
import dircache
from math import ceil

sys.path.append( "../tools/" )
from parse_out_email_text import parseOutText

"""
    starter code to process the emails from Sara and Chris to extract
    the features and get the documents ready for classification

    the list of all the emails from Sara are in the from_sara list
    likewise for emails from Chris (from_chris)

    the actual documents are in the Enron email dataset, which
    you downloaded/unpacked in Part 0 of the first mini-project

    the data is stored in lists and packed away in pickle files at the end

"""

if False:
    maildir = os.path.join('..', 'tools', 'enron_mail_20110402/maildir')
    
    dirlist = dircache.listdir(maildir)
    
    strip_words = ["sara", "shackleton", "chris", "germani", "sshacklensf", "cgermannsf", "nonprivilegedpst", "enron", "com", "the", "i", "on", "to", "a", "it", "and", "or", "in", "do", "at","fyi","not", "is", "as", "be","you", "are"]
    special_keywords = ["arthur", "anderson", "stock", "fraud", "california", "power", "grid", "sec", "president", "bush", "prosecutor", "cheney", "bushcheney","jail","prison","indicted","risk"]
    
    all_authors_keyword_count = {}
    
    
    all_word_data = []
    
    for d in dirlist:
    
        emailaliaslist = d.split('-')
        emailalias = emailaliaslist[1] + emailaliaslist[0]
        strip_words.append(d)
        strip_words.append(emailalias)
        
        word_data = []
        all_authors_keyword_count[d] = 0
        
        path = os.path.join(maildir,d)
        
        if ( not os.path.isdir(path) ):
            continue
        
        subdirlist = dircache.listdir(path)
    
        for sd in subdirlist:
            emailpath = os.path.join(path,sd)
            if (not os.path.isdir(emailpath) ) :
                continue
            
            emaillist = dircache.listdir(emailpath)
            
            temp_counter = 0
            for emailfile in emaillist:
                
                ### temp_counter is a way to speed up the development--there are
                ### thousands of emails from Sara and Chris, so running over all of them
                ### can take a long time
                ### temp_counter helps you only look at the first 200 emails in the list
                
                temp_counter += 1
                if True:
#                if temp_counter < 200:
    
                    emailfilepath = os.path.join(emailpath,emailfile)
                    
                    if ( not os.path.isfile(emailfilepath)):
                        continue
                    
                    print emailfilepath
                    email = open(emailfilepath, "r")
                           
                    ### use parseOutText to extract the text from the opened email
                    etext = parseOutText(email)
                    etext = etext.split()
                    etext2 = set()

                    for w in etext:
                        if w not in strip_words and w not in etext2:
                            etext2.add(w)

                    for w in etext2:
                        if w in special_keywords:
                            all_authors_keyword_count[d] += 1

                    etext = ' '.join([w for w in etext2])

                    ### append the text to word_data
                    word_data.append(etext)
                    all_word_data.append(etext)
    
                    email.close()
        
        print d + " emails processed"
        userwordfile = os.path.join("user_words",d + "_word_data.pkl")
        pickle.dump( word_data, open(userwordfile, "w") )
        pickle.dump( all_authors_keyword_count, open("your_authors_keyword_count.pkl","w") )
        pickle.dump( all_word_data, open("your_word_data.pkl", "w") )
    
### load in the dict of dicts containing all the data on each person in the dataset
word_data = pickle.load( open("your_word_data.pkl", "r") )
word_data_len = int(len(word_data)/2)
word_data2 = word_data[word_data_len:]
word_data = word_data[:word_data_len]

#===============================================================================
# 
# word_data = set()
# 
# for i in xrange(len(word_data2)):
#     words = word_data2[i].split()
#     for w in words:
#         if w not in word_data:
#             word_data.add(w)
#             
# word_data = list(word_data)
#===============================================================================
### in Part 4, do TfIdf vectorization here

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(word_data)
#print len(X)
pickle.dump( X, open("tfidfvector.pkl","w"))
Y = vectorizer.get_feature_names()
pickle.dump( Y, open("tfidfvector_featurenames.pkl","w"))
print len(Y)
print Y[34597]

vectorizer = TfidfVectorizer(stop_words='english')
X2 = vectorizer.fit_transform(word_data2)
#print len(X)
pickle.dump( X2, open("tfidfvector2.pkl","w"))
Y2 = vectorizer.get_feature_names()
pickle.dump( Y2, open("tfidfvector_featurenames2.pkl","w"))
print len(Y2)
print Y2[34597]