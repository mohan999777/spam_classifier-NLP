import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer


ps = PorterStemmer()

wordnet = WordNetLemmatizer()


df = pd.read_csv('/home/mohan/spam_classifier', sep ='\t', names=['labels', 'message'] )


corpus =[]

for i in range(0, len(df)):
    review = re.sub('[^a-zA-Z]', ' ', df['message'][i])   # removing symbols because they are not required for sentiment analysis 
    review = review.lower()
    review = review.split()

    review = [wordnet.lemmatize(words) for words in review  if not words in set(stopwords.words('english'))]
    review = ' '.join(review)
#     print(review)
    corpus.append(review)
    
    
#print(corpus)    

y = pd.get_dummies(df['labels'])

y = y.iloc[:,1].values


from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

X = cv.fit_transform(corpus).toarray()

print(X)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

from sklearn.naive_bayes import MultinomialNB

spam_detect_model = MultinomialNB().fit(X_train, y_train)

y_pred = spam_detect_model.predict(X_test)


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

