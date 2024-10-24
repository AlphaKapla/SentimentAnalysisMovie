import os.path as op
import re 
import numpy as np
import matplotlib.pyplot as plt
import gdown 
from glob import glob
from sklearn.model_selection import train_test_split

# download data then tar -xvf /content/aclImdb_v1.tar.gz to unarchive
#gdown.download("https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz", output="aclImdb_v1.tar.gz", quiet=False)

# We get the files from the path: ./aclImdb/train/neg for negative reviews, and ./aclImdb/train/pos for positive reviews
train_filenames_neg = sorted(glob(op.join('.', 'aclImdb', 'train', 'neg', '*.txt')))
train_filenames_pos = sorted(glob(op.join('.', 'aclImdb', 'train', 'pos', '*.txt')))

"""
test_filenames_neg = sorted(glob(op.join('.', 'aclImdb', 'test', 'neg', '*.txt')))
test_filenames_pos = sorted(glob(op.join('.', 'aclImdb', 'test', 'pos', '*.txt')))
"""

# Each files contains a review that consists in one line of text: we put this string in two lists, that we concatenate
train_texts_neg = [open(f, encoding="utf8").read() for f in train_filenames_neg]
train_texts_pos = [open(f, encoding="utf8").read() for f in train_filenames_pos]
train_texts = train_texts_neg + train_texts_pos

"""
test_texts_neg = [open(f, encoding="utf8").read() for f in test_filenames_neg]
test_texts_pos = [open(f, encoding="utf8").read() for f in test_filenames_pos]
test_texts = test_texts_neg + test_texts_pos
"""

# The first half of the elements of the list are string of negative reviews, and the second half positive ones
# We create the labels, as an array of [1,len(texts)], filled with 1, and change the first half to 0
train_labels = np.ones(len(train_texts), dtype=int)
train_labels[:len(train_texts_neg)] = 0.

"""
test_labels = np.ones(len(test_texts), dtype=np.int)
test_labels[:len(test_texts_neg)] = 0.
"""

#example of one document:
#open("./aclImdb/train/neg/0_3.txt", encoding="utf8").read()

# This number of documents may be high for most computers: we can select a fraction of them (here, one in k)
# Use an even number to keep the same number of positive and negative reviews
k = 10
train_texts_reduced = train_texts[0::k]
train_labels_reduced = train_labels[0::k]

print('Number of documents:', len(train_texts_reduced))

train_texts_splt, val_texts, train_labels_splt, val_labels = train_test_split(train_texts_reduced, train_labels_reduced, test_size=.2)

# Create and fit the vectorizer to the training data
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
train_bow = vectorizer.fit_transform(train_texts_splt)

print(train_bow.shape)

# Transform the validation data
validation_bow = vectorizer.transform(val_texts) 

print(validation_bow.shape)
from sklearn.naive_bayes import MultinomialNB
X = train_bow.toarray()
print("X shape",X.shape)
Y = train_labels_splt
print("Y shape",Y.shape)
clf = MultinomialNB()
clf.fit(X, Y)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# Test it on the validation data 
validation_predictions = clf.predict(validation_bow.toarray())

# Calculate the confusion matrix
cm = confusion_matrix(val_labels, validation_predictions)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot()

# Keep the plot window open
# plt.show() 

# Print the classification report
print(classification_report(val_labels, validation_predictions))

# get feature names
# print(vectorizer.get_feature_names_out()[:100])

from sklearn.pipeline import Pipeline

pipeline_base = Pipeline([
    ('vect', CountVectorizer(max_features=30000, analyzer='word', stop_words=None)),
    ('clf', MultinomialNB())])

print("score is monogram =",pipeline_base.fit(train_texts_splt, train_labels_splt).score(val_texts, val_labels))

# Fit and test a pipeline with bigrams,

pipeline_base_bigram = Pipeline([
    ('vect', CountVectorizer(max_features=30000, ngram_range=(1, 2), analyzer='word', stop_words=None)),
    ('clf', MultinomialNB())])

print("score is when adding bigram =",pipeline_base_bigram.fit(train_texts_splt, train_labels_splt).score(val_texts, val_labels))

# ... trigrams,

pipeline_base_trigram = Pipeline([
    ('vect', CountVectorizer(max_features=30000, ngram_range=(1, 3), analyzer='word', stop_words=None)),
    ('clf', MultinomialNB())])

print("score is when adding trigram =",pipeline_base_trigram.fit(train_texts_splt, train_labels_splt).score(val_texts, val_labels))

# ... and characters.

pipeline_base_trigram_char = Pipeline([
    ('vect', CountVectorizer(max_features=30000, ngram_range=(1, 3), analyzer='word', stop_words=None, max_df=0.5)),
    ('clf', MultinomialNB())])

print("score is when adding trigram and dfmax =",pipeline_base_trigram_char.fit(train_texts_splt, train_labels_splt).score(val_texts, val_labels))

## Tf-idf

from sklearn.feature_extraction.text import TfidfTransformer

# Fit and test a pipeline with tf-idf

pipetfidf = Pipeline([
            ('vect', CountVectorizer(max_features=30000, ngram_range=(1, 2), analyzer='word', stop_words=None, max_df=0.5)),
            ('tfid', TfidfTransformer()),
            ('clf', MultinomialNB())]).fit(train_texts_splt)



#print("score is with tf-idf =",pipeline_tfidf.fit(train_texts_splt, train_labels_splt).score(val_texts, val_labels))
