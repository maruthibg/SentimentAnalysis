# https://www.tensorscience.com/nlp/sentiment-analysis-tutorial-in-python-classifying-reviews-on-movies-and-products
# https://www.analyticsvidhya.com/blog/2018/02/the-different-methods-deal-text-data-predictive-python/
# https://www.kdnuggets.com/2018/03/text-data-preprocessing-walkthrough-python.html

import re
import contractions
import unicodedata
import inflect

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer
from nltk.stem import LancasterStemmer

def openFile(path, type='csv'):
    #param path: path/to/file.ext (str)
    #Returns contents of file (str)
    if type == 'csv':
        with open(path) as file:
            data = file.read()
    if type == 'excel':
        data  = pd.read_excel(path)
    return data

def replace_contractions(text):
    """Replace contractions in string of text"""
    return contractions.fix(text)

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    custom_stopwords = ['hi', 'hello', 'thanks', 'thank', 'warm', 'regards', 'regard', 'best', 'kind', 'dear', 'br', 'kindly', 'rohini', 'jigar', 'dillo']
    stop_words = stopwords.words('english')#
    #for i in custom_stopwords:
    #    stop_words.append(i)

    new_words = []
    for word in words:
        if word not in stop_words:
            new_words.append(word)
    return new_words

def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def join_words(words):
    return ' '.join(words)

def normalize(sample):
    words = nltk.word_tokenize(sample)
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    words = remove_stopwords(words)
    words = stem_words(words)
    words = join_words(words)
    return words

def dataframe():
    df = openFile(r'D:\PROJECTS\machinelearning\projects\sentiment-analysis\output.xlsx', type='excel')
    df = df.sample(frac=1)
    pd.value_counts(df['Label']).plot.bar()
    plt.title('Label histogram')
    plt.xlabel('Label')
    plt.ylabel('Count')
    df['Label'].value_counts()
    return df

df = dataframe()

df['Comment'] = df['Comment'].apply(lambda x: x.lstrip("b"))
df['Comment'] = df['Comment'].apply(replace_contractions)
df['Comment'] = df['Comment'].apply(normalize)

#
df.replace('', np.NaN, inplace=True)
df.dropna(inplace=True)

#Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
bow = vectorizer.fit_transform(df['Comment'])
len(vectorizer.get_feature_names())
labels = df['Label']


from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score, StratifiedKFold, \
     train_test_split
X_train, X_test, y_train, y_test = train_test_split(bow, labels, test_size=0.33)

print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions X_test dataset: ", X_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)

print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))

"""
sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

clasifiers = [
    LogisticRegression(),
    #KNeighborsClassifier(3),
    #SVC(kernel="rbf", C=0.025, probability=True),
    #NuSVC(probability=True),
    #LinearSVC(),
    #DecisionTreeClassifier(),
    #RandomForestClassifier(),
    #AdaBoostClassifier(),
    #GradientBoostingClassifier(),
    #GaussianNB(),
    #MultinomialNB(),
    #LinearDiscriminantAnalysis(),
    #QuadraticDiscriminantAnalysis()
    ]

# Logging for Visual Comparison
log_cols=["Classifier", "Accuracy", "Log Loss"]
log = pd.DataFrame(columns=log_cols)

for clf in classifiers:
    clf.fit(X_train, y_train)
    name = clf.__class__.__name__
    
    print("="*30)
    print(name)
    
    print('****Results****')
    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    print("Accuracy: {:.4%}".format(acc))
    
    train_predictions = clf.predict_proba(X_test)
    ll = log_loss(y_test, train_predictions)
    print("Log Loss: {}".format(ll))
    
    log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)
    log = log.append(log_entry)
    
    kfold = StratifiedKFold(n_splits=10, random_state=10)
    score = cross_val_score(clf, X_train, y_train, cv=kfold, scoring='accuracy').mean()
    print("Accuracy: {:.4%}".format(score))
    
print("="*30)


comment = ["any update ?"]
print("Prediction: {}". format(clf.predict(vectorizer.transform(comment))))

comment = ["Thank you"]
print("Prediction: {}". format(clf.predict(vectorizer.transform(comment))))
