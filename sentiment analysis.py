#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import the necesdsary libraries
# warnings: ignore all the warnings
# pandas: for dealing with the csv file
# seaborn: for data visualization
# scikit-learn: for building the model
# matplotlib: also for visualization

import warnings 
warnings.filterwarnings('ignore') 
import pandas as pd 
import re 
import seaborn as sns 
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer 
import matplotlib.pyplot as plt 
from wordcloud import WordCloud


# In[2]:


# We will also use nltk for text analysis
# we also need the stopword and punkt

import nltk 
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords


# In[3]:


# import the dataset
# Show the first 20 reviews & corresponding ratings

data = pd.read_csv('flipkart_data.csv') 
data.head(20)


# In[4]:


# Show the statistical summary of the data

data.describe()


# In[5]:


# Let's see how many unique ratings do we have

pd.unique(data['rating'])


# In[6]:


# Let's visualize the unique ratings using bar chart

sns.countplot(data=data, x='rating', order=data.rating.value_counts().index)


# In[7]:


# Now, we want to predict whether the sentiment is a positive comment or negative comments
# so we need to change the rating column into a column of 0s and 1s
# 0s means negative comments
# 1s means positive comments

# Let's treat those ratings with rating 4 or above as positive
# while rating 3 or below as negative


positive_negative = [] 
for i in range(len(data['rating'])): 
    if data['rating'][i] >= 4: 
        positive_negative.append(1) 
    else: 
        positive_negative.append(0) 

data['label'] = positive_negative 


# In[8]:


# This is the function for pre-processing the data

from tqdm import tqdm 


def preprocess_text(text_data): 
    preprocessed_text = [] 

    for sentence in tqdm(text_data): 
        # Removing punctuations 
        sentence = re.sub(r'[^\w\s]', '', sentence) 

        # Converting lowercase and removing stopwords 
        preprocessed_text.append(' '.join(token.lower() 
                                          for token in nltk.word_tokenize(sentence) 
                                          if token.lower() not in stopwords.words('english'))) 

    return preprocessed_text


# In[9]:


# Let's process the dataset now

preprocessed_review = preprocess_text(data['review'].values) 
data['review'] = preprocessed_review


# In[10]:


# After the dataset is processed, let's see the first 20 rows of data

data.head(20)


# In[11]:


# Show the statistical summary of the processed data

data.describe()


# In[12]:


# Now, let's count how many positive comments & negative comments respectively

data["label"].value_counts()


# In[13]:


# Let's also create a wordcloud of those comments with labe 1 (i.e. positive comments)

consolidated = ' '.join(word for word in data['review'][data['label'] == 1].astype(str)) 

wordCloud = WordCloud(width=1600, height=800, random_state=21, max_font_size=110)

plt.figure(figsize=(15, 10)) 
plt.imshow(wordCloud.generate(consolidated), interpolation='bilinear') 
plt.axis('off') 
plt.show() 


# In[14]:


# let's vectorize the words (vectorization)

cv = TfidfVectorizer(max_features=2500) 
X = cv.fit_transform(data['review'] ).toarray()

# TF-IDF calculates that how relevant a word in a series or corpus is to a text. 
# The meaning increases proportionally to the number of times in the text a word appears 
# but is compensated by the word frequency in the corpus (data-set).


# In[15]:


print(X)


# In[16]:


# We do train-test split first, and then train a model

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, data['label'], test_size=0.33, stratify=data['label'], random_state = 42)


# In[17]:


# Use decision-tree for prediction

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

model = DecisionTreeClassifier(random_state=0) 
model.fit(X_train, y_train) 

# Test the model 
pred = model.predict(X_train) 
print("Accuracy: ", accuracy_score(y_train,pred))


# In[18]:


# Show the confusion matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train,pred) 

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [False, True]) 

cm_display.plot() 
plt.show()

