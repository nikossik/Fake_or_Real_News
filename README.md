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










