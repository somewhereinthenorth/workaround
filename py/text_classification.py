#Auf Basis des Tutrials von https://stackabuse.com/text-classification-with-python-and-scikit-learn/
import numpy as np  
import re  
import nltk  
from sklearn.datasets import load_files  
nltk.download('all-nltk') 
nltk.download('stopwords')  
import pickle  
from nltk.corpus import stopwords  
#Datensets laden
movie_data = load_files(r"datasets\txt_sentoken")  #Relativer Pfad
X, y = movie_data.data, movie_data.target  
#Preprozessor
documents = []
from nltk.stem import WordNetLemmatizer
stemmer = WordNetLemmatizer()
#startloop
for sen in range(0, len(X)):  
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(X[sen]))
    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)
    # Converting to Lowercase
    document = document.lower()
    # Lemmatization
    document = document.split()
    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)
    documents.append(document)
#endloop

#Wörter in Zahlen umwandeln
from sklearn.feature_extraction.text import CountVectorizer  
vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))  
#max_features: Maximale Anzahl an Wörtern, die am häufigsten aufgetreten sind.
#min_df: 
#stop_words: Wörter, die in Texten häufig Verwendungfinden, jedoch zu dem eigentlichen Informationsgehalt eines Textes nichts beitragen. Zum Beispiel "das", dieses Wort hat nur eine grammatikalische Funktion.
X = vectorizer.fit_transform(documents).toarray()  

