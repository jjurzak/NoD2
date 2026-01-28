import pandas as pd 
from pandas.core.computation.parsing import token
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from gensim.models import Word2Vec, LdaModel
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt 
from gensim.corpora import Dictionary
import string

df = pd.read_csv("IMDB_Dataset.csv")


documents = df["review"].tolist()

#print(documents)

vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words="english"
        )

X_tfidf = vectorizer.fit_transform(documents)

#print(X_tfidf.shape)

svd = TruncatedSVD(n_components=2, random_state=2137)

X_2d = svd.fit_transform(X_tfidf)

print(X_2d.shape)

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))
punctuation = set(string.punctuation)

tokenized_docs = [
        word_tokenize(doc.lower())
        for doc in documents
        ]

dictionary = Dictionary(tokenized_docs)

cleaned_docs = []

for doc in tokenized_docs:
    cleaned = [
            token
            for token in doc
            if token.isalpha()
            and token not in stop_words
            and len(token) > 2
            ]
    cleaned_docs.append(cleaned)

dictionary = Dictionary(cleaned_docs)
print(cleaned_docs[0][:20])

corpus = [dictionary.doc2bow(doc) for doc in cleaned_docs]

lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=5,
        passes=10,
        random_state=2137
        )

for idx, topic in lda_model.print_topics(num_words=10):
    print(f'Temat {idx}: {topic}')

#print(list(dictionary.items())[:10])

sentences = cleaned_docs

w2v_model = Word2Vec(
        sentences=sentences,
        vector_size=100,
        window=5,
        min_count=5,
        workers=4,
        seed=2137
        )

print("similarity(good, great):", w2v_model.wv.similarity("good", "great"))
print("most similar to 'good':", w2v_model.wv.most_similar("good"))
print("most similar to 'horror':", w2v_model.wv.most_similar("horror"))
print("most similar to 'boring':", w2v_model.wv.most_similar("boring"))



plt.figure(figsize=(8,6))
plt.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.3, s=10)
plt.xlabel("Składowa x")
plt.ylabel("Składowa y")
plt.title("Reprezentacja dokumentow w 2D (TF-IDF + SVD)")
plt.show()



