import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.decomposition import TruncatedSVD,LatentDirichletAllocation

# Phase 1:   Laden der Daten

data_set = pd.read_csv("data\\airfrance_tripadvisor_reviews.csv")
corpus = data_set["text"].tolist()

# Phase 2:  Bereinigung der Daten

# Herunterladen der Stopwörter
nltk.download("stopwords")
stopwords_list = stopwords.words("english")

reviews = []
# Tokenisierung, Umwandlung in Kleinbuchstaben und Filterung nach Buchstaben

for r in corpus:
  x = word_tokenize(r)
  rev = [w.lower() for w in x if w.isalpha()]
  
  # Filterung nach Stopwörtern
  
  rev_clean = [t for t in rev if t not in stopwords_list]

  # Erneutes Zusammenfügen der bereinigten Reviews
  
  reviews.append(" ".join(rev_clean))

# Phase 3: Vektorisierung

# BoW 
bow_vect = CountVectorizer(max_features=1000)
bow_mtrx = bow_vect.fit_transform(reviews)

# Ausgabe der Top-Wörter aus der BoW
print("\n---- Top Wörter aus der BoW ----\n")
wort_summen_bow = bow_mtrx.sum(axis=0).A1
top_words_bow = bow_vect.get_feature_names_out()
df = pd.DataFrame({'Wort': top_words_bow, 'Score': wort_summen_bow})
df_sorted = df.sort_values(by='Score', ascending=False)
print(df_sorted.head(10))

# TF-IDF
tfidf_vect = TfidfVectorizer(use_idf=True, max_features=1000, smooth_idf=True)
tfidf_mtrx = tfidf_vect.fit_transform(reviews)

# Ausgabe der Top-Wörter aus der TF-IDF
print("\n---- Top Wörter aus der TF-IDF ----\n")
wort_summen_tfidf = tfidf_mtrx.sum(axis=0).A1
top_words_tfidf = tfidf_vect.get_feature_names_out()
df = pd.DataFrame({'Wort': top_words_tfidf, 'Score': wort_summen_tfidf})
df_sorted = df.sort_values(by='Score', ascending=False)
print(df_sorted.head(10))

# Phase 4: Themen finden

# LSA mit TF-IDF
print("\n---- Top Themen aus LSA mit TF-IDF ----\n")
LSA_model = TruncatedSVD(n_components=5, algorithm="randomized", n_iter=10)
lsa = LSA_model.fit_transform(tfidf_mtrx)
for i, topic in enumerate(LSA_model.components_):
  indizes = list(topic.argsort())
  indizes.reverse()
  top_words = []
  for j in indizes[:5]:
    top_words.append(top_words_tfidf[j])
  print("Thema ",i, " : ", top_words)

# LDA mit BoW
print("\n---- Top Themen aus LDA mit BoW ----\n")
LDA_model = LatentDirichletAllocation(n_components=5, learning_method="online", random_state=42, max_iter=10)
lda = LDA_model.fit_transform(bow_mtrx)

for i, topic in enumerate(LDA_model.components_):
  indizes = list(topic.argsort())
  indizes.reverse()
  top_words = []
  for j in indizes[:5]:
    top_words.append(top_words_bow[j])
  print("Thema ",i," : ", top_words)
