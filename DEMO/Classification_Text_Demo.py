import os
import random
import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk import word_tokenize
from collections import defaultdict
from nltk import FreqDist
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import pickle

stop_words = set(stopwords.words('english'))
stop_words.add('said')
stop_words.add('mr')

BASE_DIR ='D:\\Learning\\School\\Semester__6\\COMPUTER_SCIENCE_PROJECT_2\\DEMO\\bbc'
LABELS = ['business', 'entertainment', 'politics', 'sport', 'tech']

def create_data_set():
    with open('data.txt', 'w', encoding='utf8') as outfile:
        for label in LABELS:
            dir = '%s/%s' % (BASE_DIR, label)
            for filename in os.listdir(dir):
                fullfilename = '%s/%s' % (dir, filename)
                print(fullfilename)
                with open(fullfilename, 'rb') as file:
                    text = file.read().decode(errors= 'replace').replace('\n', '')
                    outfile.write('%s\t%s\t%s\n' % (label, filename, text))

def setup_docs():
    docs = [] # Contain 'text' - 'label'
    with open('DEMO/data.txt', 'r', encoding='utf8') as datafile:
        for row in datafile:
            parts = row.split('\t') 
            doc = ( parts[0], parts[2].strip() ) 
            docs.append(doc) 
        return docs 

def clean_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    return text

def get_tokens(text):
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if not t in stop_words]
    return tokens

def print_frequency_dist(docs):
    tokens = defaultdict(list)
    
    for doc in docs:
        doc_label = doc[0]
        
        doc_text = clean_text(doc[1])
        
        doc_tokens = get_tokens(doc_text)
        
        tokens[doc_label].extend(doc_tokens)
    for category_label, category_tokens in tokens.items():
        print(category_label)
        fd = FreqDist(category_tokens)
        print(fd.most_common(5))

def get_splits(docs):
    
    random.shuffle(docs)
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    
    pivot = int(.80 * len(docs))
    
    for i in range(0, pivot):
        X_train.append(docs[i][1]) 
        y_train.append(docs[i][0])
    
    for i in range(pivot, len(docs)):
        X_test.append(docs[i][1]) 
        y_test.append(docs[i][0])
    
    return X_train, X_test, y_train, y_test

def evaluate_classifier(title, classifier, vectorizer, X_test, y_test):
    
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = classifier.predict(X_test_tfidf)

    precision = metrics.precision_score(y_test, y_pred, average='macro') 
    recall = metrics.recall_score(y_test, y_pred, average='macro') 
    f1 = metrics.f1_score(y_test, y_pred, average='macro') 

    print("%s\t%f\t%f\t%f\n" % (title, precision, recall, f1)) 

def train_classifier(docs):
    X_train, X_test, y_train, y_test = get_splits(docs) 
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 3), min_df=3, analyzer='word')
    
    dtm = vectorizer.fit_transform(X_train)

    naive_bayes_classifier = MultinomialNB().fit(dtm, y_train)
    
    evaluate_classifier("Naive Bayes\tTRAIN\t", naive_bayes_classifier, vectorizer, X_train, y_train)
    evaluate_classifier("Naive Bayes\tTEST\t", naive_bayes_classifier, vectorizer, X_test, y_test)
    
    clf_filename = 'naive_bayes_classifier.pkl'
    pickle.dump(naive_bayes_classifier, open(clf_filename, 'wb'))
    vec_filename = 'count_vectorizer.pkl'
    pickle.dump(vectorizer, open(vec_filename, 'wb'))

def classify(text):
   
    clf_filename = 'D:\\Learning\School\\Semester__6\\COMPUTER_SCIENCE_PROJECT_2\\DEMO\\naive_bayes_classifier.pkl'
    nb_clf = pickle.load(open(clf_filename, 'rb'))
    vec_filename = 'D:\\Learning\School\\Semester__6\\COMPUTER_SCIENCE_PROJECT_2\\DEMO\\count_vectorizer.pkl'
    
    vectorizer = pickle.load(open(vec_filename, 'rb'))
    pred = nb_clf.predict(vectorizer.transform([text]))
    
    print(f"\n\nVăn bản được phân loại: {str(pred[0]).upper()}\n\n")

if __name__ == '__main__':
    #create_data_set()
    
    #docs = setup_docs()
    
    #print_frequency_dist(docs)
    
    #train_classifier(docs)
    
    new_doc = """We are continuing to work closely with German police to support a safe and trouble-free Euro 2024 tournament for travelling England fans.”

Up 40,000 England fans and 8,000 Serbia fans are expected to travel to Gelsenkirchen for a game UEFA, European football’s governing body, and German police have designated as “high-risk.”

In total, some 300,000 England fans are expected to travel to Germany during the tournament. CNN understands the British police have around 2,000 banning orders currently in place for fans."""
    
    classify(new_doc)
    
    print("Hoàn thành")