# Fake or Real News 
Classifying the news

![news](https://s.rbk.ru/v5_marketing_media/images/5/64/116049080264645.jpg)

## [Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
### In this competition, our task is to classify the news into true and false
--- 

## IMPORT LIBRARY & PACKAGES

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.tokenize import word_tokenize,sent_tokenize
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
```
--- 
## EXPLORATORY DATA ANALYSIS

##### Reading data
```python
fake = pd.read_csv('/content/drive/MyDrive/Fake_True_News/archive/Fake.csv')
true = pd.read_csv('/content/drive/MyDrive/Fake_True_News/archive/True.csv')
```
```python
print(true.shape)
print(true.info())
```
```
output:
(21417, 4)
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 21417 entries, 0 to 21416
Data columns (total 4 columns):
 #   Column   Non-Null Count  Dtype 
---  ------   --------------  ----- 
 0   title    21417 non-null  object
 1   text     21417 non-null  object
 2   subject  21417 non-null  object
 3   date     21417 non-null  object
dtypes: object(4)
memory usage: 669.4+ KB
```

```python
print(fake.shape)
print(fake.info())
```
```
output:
(23481, 4)
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 23481 entries, 0 to 23480
Data columns (total 4 columns):
 #   Column   Non-Null Count  Dtype 
---  ------   --------------  ----- 
 0   title    23481 non-null  object
 1   text     23481 non-null  object
 2   subject  23481 non-null  object
 3   date     23481 non-null  object
dtypes: object(4)
memory usage: 733.9+ KB
```
Then we create data form fake and true (pd.concat):
```python
data = pd.concat([true,fake],axis=0,ignore_index=True)
print(data.shape)
```
```
output:
(44898, 5)
```
Let's look at the distribution of 1 and 0
```python
sns.countplot(data.Label)
```
![out](/photo/10.png)
---
## DATA CLEANING
#### Lowercase words, remove the word 'Reuters', remove square brackets, links, words containing numbers and punctuations
- Cleaning our text data is important so that the model wont be fed noises that would not help with the prediction.
- The word reuters was removed as it always appear in the real news article therefore I removed it as it is an obvious indicator to the model

```python
def clean_text(text):
    
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('Reuters','',text)
    return text
```
#### Remove stop words:
```python
stop = stopwords.words('english')
data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
```

#### Lemmatize words
Words were lemmatized so that only root words are retain in the data and fed into the model

```python
def lemmatize_words(text):
    wnl = nltk.stem.WordNetLemmatizer()
    lem = ' '.join([wnl.lemmatize(word) for word in text.split()])    
    return lem
```
#### Split data into train and test set
```python
X_train, X_test, y_train, y_test = train_test_split(data['text'], y,test_size=0.33,random_state=53)
```

#### Using Bag of words model for data transformation
Since we are dealing with text data, we cannot fed it directly to our model. Therefore, I am using bag of words model to extract features from our text data and convert it into numerical feature vectors that can be fed directly to the algorithm

```python
count_vectorizer = CountVectorizer(stop_words='english')
count_train = count_vectorizer.fit_transform(X_train.values)
count_test = count_vectorizer.transform(X_test.values)
print(count_train.shape)
```
```
(30081, 172639)
```
--- 
## MODEL
Using 2 different model with different parameter for parameter investigation values of alpha and c 

#### Naive Bayes

```python
from sklearn.metrics import classification_report

nb_classifier = MultinomialNB()
nb_classifier.fit(count_train, y_train)

pred = nb_classifier.predict(count_test)

print(classification_report(y_test, pred, target_names = ['Fake','True']))
```
```
              precision    recall  f1-score   support

        Fake       0.95      0.96      0.95      7178
        True       0.96      0.95      0.96      7639

    accuracy                           0.96     14817
   macro avg       0.95      0.96      0.96     14817
weighted avg       0.96      0.96      0.96     14817
```
#### Support Vector Machine (SVM)
```python
from sklearn.svm import SVC

svc_model = SVC(C=1, kernel='linear', gamma= 1)
svc_model.fit(count_train, y_train)

prediction1 = svc_model.predict(count_test)

print(classification_report(y_test, prediction1, target_names = ['Fake','True']))
```
```
              precision    recall  f1-score   support

        Fake       0.99      0.99      0.99      7178
        True       0.99      0.99      0.99      7639

    accuracy                           0.99     14817
   macro avg       0.99      0.99      0.99     14817
weighted avg       0.99      0.99      0.99     14817
```
