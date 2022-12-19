import numpy as np
import pandas as pd

df=pd.read_csv('./spam.csv', encoding='ISO-8859-1')

df.sample(5)

df.shape

# 1. Data Cleaning
# 2. EDA (Exploratory Data Analysis)
# 3. Text Preprocessing
# 4. Model Building
# 5. Evaluation
# 6. Improvement
# 7. Website
# 8. Deploy


#------------Data Cleaning-----------------

#drop last 3 columns
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'], inplace=True)
df.sample(5)

#rename columns
df.rename(columns={'v1':'target', 'v2':'text'}, inplace=True)



from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()

df['target']=encoder.fit_transform(df['target'])
df.head()

#missing values
df.isnull().sum()

#check for duplicate values
df.duplicated().sum()
df=df.drop_duplicates(keep='first')
df.duplicated().sum()
print(df.shape)


#-------------------EDA-----------
df.head()
df['target'].value_counts()

import matplotlib.pyplot as plt
plt.figure(1)
plt.pie(df['target'].value_counts(), labels=['ham','spam'], autopct="%0.2f")

#Data is imbalanced
import nltk

# nltk.download('punkt')
df['num_characters']=df['text'].apply(len)
df.head()

#number of words
df['num_words'] = df['text'].apply(lambda x: len(nltk.word_tokenize(x)))

df.head()

df['num_sentences'] = df['text'].apply(lambda x: len(nltk.sent_tokenize(x)))

df[['num_characters','num_words','num_sentences']].describe()

ham = df[df['target'] == 0]
spam = df[df['target'] == 1]

ham[['num_characters', 'num_words', 'num_sentences']].describe()
spam[['num_characters', 'num_words', 'num_sentences']].describe()

import seaborn as sns

plt.figure(2)
sns.histplot(ham['num_characters'])
sns.histplot(spam['num_characters'])

plt.figure(3)
sns.histplot(ham['num_words'])
sns.histplot(spam['num_words'])

plt.figure(4)
sns.pairplot(df, hue='target')

# plt.figure(5)
# sns.heatmap(df.corr(), annot=True)
# plt.show()

#-----------Data Preprocessing------------
# lowercase, tokenization, remove special characters, removing stopwords and punctuation, stemming
from nltk.corpus import stopwords
# nltk.download('stopwords')
import string
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)

    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)

    text=y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text=y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))


    return " ".join(y)

df['transformed_text']=df['text'].apply(transform_text)
df.head()

spam_corpus=[]

for msg in df[df['target'] == 1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)

len(spam_corpus)

from collections import Counter
pd.DataFrame(pd.DataFrame(Counter(spam_corpus).most_common(30)))

#---------------Model Building----------------- 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
cv =CountVectorizer()
tfidf=TfidfVectorizer()

X = tfidf.fit_transform(df['transformed_text']).toarray()
X.shape

y=df['target'].values
print(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score

gnb=GaussianNB()
mnb=MultinomialNB()
bnb=BernoulliNB()
# print('\n')
# gnb.fit(X_train, y_train)
# y_pred1=gnb.predict(X_test)
# print("Accuracy Score ", accuracy_score(y_test, y_pred1))
# print('-----------------------------------')
# print("Confusion Matrix \n", confusion_matrix(y_test, y_pred1))
# print('-----------------------------------')
# print("Precision Score: ", precision_score(y_test, y_pred1))
# print('\n')
mnb.fit(X_train, y_train)
y_pred2=mnb.predict(X_test)
print("Accuracy Score ", accuracy_score(y_test, y_pred2))
print('-----------------------------------')
print("Confusion Matrix \n", confusion_matrix(y_test, y_pred2))
print('-----------------------------------')
print("Precision Score: ", precision_score(y_test, y_pred2))
print('\n')
# bnb.fit(X_train, y_train)
# y_pred3=bnb.predict(X_test)
# print("Accuracy Score ", accuracy_score(y_test, y_pred3))
# print('-----------------------------------')
# print("Confusion Matrix \n", confusion_matrix(y_test, y_pred3))
# print('-----------------------------------')
# print("Precision Score: ", precision_score(y_test, y_pred3))
# print('\n')

# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier

# svc = SVC(kernel='sigmoid', gamma=1.0)
# knc = KNeighborsClassifier()
# mnb = MultinomialNB()
# dtc = DecisionTreeClassifier(max_depth=5)
# lrc = LogisticRegression(solver='liblinear', penalty='l1')

# clfs = {
#     'Support Vector Machines Classifier' : svc,
#     'K Nearest Neighbor Classifier' : knc, 
#     'Multinomial Naive Bayes Classifier': mnb, 
#     'Decision Tree Classifier': dtc, 
#     'Logistic Regression Classifier': lrc
# }

# def train_classifier(clf,X_train,y_train,X_test,y_test):
#     clf.fit(X_train,y_train)
#     y_pred = clf.predict(X_test)
#     accuracy = accuracy_score(y_test,y_pred)
#     precision = precision_score(y_test,y_pred)
    
#     return accuracy,precision


# accuracy_scores = []
# precision_scores = []

# for name,clf in clfs.items():
#     current_accuracy,current_precision = train_classifier(clf, X_train,y_train,X_test,y_test)
    
#     print("For ",name)
#     print("Accuracy - ",current_accuracy)
#     print("Precision - ",current_precision)
    
#     accuracy_scores.append(current_accuracy)
#     precision_scores.append(current_precision)
#     print('\n')

# performance_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy':accuracy_scores,'Precision':precision_scores}).sort_values('Precision',ascending=False)

# print(performance_df)
# print('\n')

mail_body="free"
t=transform_text(mail_body)
v=tfidf.transform([t])
result=mnb.predict(v)
print(result)