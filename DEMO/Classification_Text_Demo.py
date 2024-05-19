import tkinter as tk
import random
import string
from nltk import word_tokenize
from collections import defaultdict
from nltk import FreqDist
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import pickle
from Create_Dataset import create_data_set

stop_words = set(stopwords.words('english'))
stop_words.add('said')
stop_words.add('mr')

def setup_docs():
    docs = []
    with open('data.txt', 'r', encoding='utf8') as datafile:
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
        #doc_text = doc[1] sau khi đã tìm ra các từ xuất hiện nhiều nhất, đến bước clean text #1
        doc_text = clean_text(doc[1]) # clean text, xóa bỏ các dấu câu #2
        #doc_tokens = word_tokenize(doc_text) #3
        doc_tokens =get_tokens(doc_text) #4
        tokens[doc_label].extend(doc_tokens)
    for category_label, category_tokens in tokens.items():
        print(category_label)
        fd = FreqDist(category_tokens)
        print(fd.most_common(20))

def get_splits(docs):
    random.shuffle(docs)
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    pivot = int(.80 * len(docs)) # tính đoán index của điểm chia tách giữa tập huấn luyện và tập kiểm tra (80% train và 20% test)
    for i in range(0, pivot):
        X_train.append(docs[i][1]) # Lặp qua các bộ dữ liệu từ đầu đến điểm chia tách (80%), thêm dữ liệu và nhãn tương ứng vào tập huấn luyện
        y_train.append(docs[i][0])
    for i in range(pivot, len(docs)):
        X_test.append(docs[i][1]) # Lặp qua các bộ dữ liệu từ điểm chia tách đến cuối danh sách (20%), thêm dữ liệu và nhãn tương ứng vào tập kiểm tra 
        y_test.append(docs[i][0])
    return X_train, X_test, y_train, y_test

# đánh giá hiệu suất của một mô hình phân loại trên tập dữ liệu kiểm tra
def evaluate_classifier(title, classifier, vectorizer, X_test, y_test):
    X_test_tfidf = vectorizer.transform(X_test) # Biến đổi tập dữ liệu kiểm tra X_test thành ma trận TF-IDF (đảm bảo đầu vào cho mô hình có ma trận phù hợp)
    y_pred = classifier.predict(X_test_tfidf) # Sử dụng mô hình phân loại classifier để dự đoán nhãn cho dữ liệu kiểm tra

    precision = metrics.precision_score(y_test, y_pred, average='macro') # Tính toán độ chính xác của mô hình dự đoán
    recall = metrics.recall_score(y_test, y_pred, average='macro') # Tính toán chỉ số Recall của mô hình
    f1 = metrics.f1_score(y_test, y_pred, average='macro') # Tính toán chỉ số F1-score của mô hình dự đoán 

    print("%s\t%f\t%f\t%f\n" % (title, precision, recall, f1)) # In ra tiêu đề của phân loại, cùng với các chỉ số đánh giá Precision, Recall và F1-score

# huấn luyện một mô hình phân loại trên tập dữ liệu được cung cấp và lưu trữ mô hình và vectorizer để sử dụng cho dữ liệu mới
def train_classifier(docs):
    X_train, X_test, y_train, y_test = get_splits(docs) # chia tập dữ liệu thành tập huấn luyện và tập kiểm tra
    # Khởi tạo một đối tượng CountVectorizer để biến đổi các văn bản thành các vectơ đặc trưng
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 3), min_df=3, analyzer='word')
    # tạo ma trận văn bản đặc trưng từ dữ liệu huấn luyện 
    dtm = vectorizer.fit_transform(X_train)
    # Khởi tạo và huấn luyện một mô hình phân loại Naive Bayes
    naive_bayes_classifier = MultinomialNB().fit(dtm, y_train)
    
    # Đánh giá hiệu suất của mô hình trên tập huấn luyện và in ra kết quả
    evaluate_classifier("Naive Bayes\tTRAIN\t", naive_bayes_classifier, vectorizer, X_train, y_train)
    # Đánh giá hiệu suất của mô hình trên tập kiểm tra và in ra kết quả
    evaluate_classifier("Naive Bayes\tTEST\t", naive_bayes_classifier, vectorizer, X_test, y_test)

    # Đặt tên cho tệp lưu trữ mô hình phân loại và lưu trữ mô hình phân loại đã được huấn luyện vào tệp
    clf_filename = 'naive_bayes_classifier.pkl'
    pickle.dump(naive_bayes_classifier, open(clf_filename, 'wb'))

    # Đặt tên cho tệp lưu trữ vectorizer và lưu
    vec_filename = 'count_vectorizer.pkl'
    pickle.dump(vectorizer, open(vec_filename, 'wb'))

# phân loại một đoạn văn bản mới vào một trong các nhãn của mô hình phân loại đã được huấn luyện
def classify(text):
    # load mô hình phân loại
    clf_filename = 'D:\\MYLEARNING\\THE_JOURNEY_IV\\COMPUTER_SCIENCE_PROJECT_2\\naive_bayes_classifier.pkl'
    nb_clf = pickle.load(open(clf_filename, 'rb'))

    # định nghĩa biến là đường dẫn tới tệp chứa vectorizer đã được lưu trữ sau đó mở và load vectorizer từ tệp đã lưu trữ gán cho biến vectorizer
    vec_filename = 'D:\\MYLEARNING\\THE_JOURNEY_IV\\COMPUTER_SCIENCE_PROJECT_2\\count_vectorizer.pkl'
    vectorizer = pickle.load(open(vec_filename, 'rb'))

    # Sử dụng vectorizer để chuyển đổi đoạn văn bản mới thành vectơ đặc trưng, sau đó sử dụng mô hình phân loại để dự đoán nhãn của đoạn văn bản này
    pred = nb_clf.predict(vectorizer.transform([text]))

    print(f"\n\nThe topic classified for the text is: {str(pred[0]).upper()}\n\n")

if __name__ == '__main__':
    #create_data_set()
    #docs = setup_docs()
    #print_frequency_dist(docs)
    #train_classifier(docs)
    new_doc = """   """
    classify(new_doc)
    print("Done")