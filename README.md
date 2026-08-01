# NLP-Techniken anwenden, um eine Textsammlung zu analysieren

 Zur Bearbeitung der Aufgabenstellung wird als Beispieldatenquelle der Datensatz „air-france-reviews-dataset“ ausgewählt. Dieser Datensatz hat eine Kaggle-Useability-Wertung von 10, ist komplett in englischer Sprache verfügbar und durch die Open Data Commons Attribution License (ODC-By) v1.0 frei nutzbar. Da es sich bei dem Datensatz um eine CSV-Datei handelt, kann hier die Einbindung über die pandas python Bibliothek realisiert werden.

Um „saubere Texte“ zu erhalten, müssen im Vorfeld Stoppwörter eingebunden werden. Diese können dann mit eigenen Wörtern angereichert werden. Anschließend werden alle Wörter in Kleinbuchstaben umgewandelt und alle nicht-alphabetischen Zeichen entfernt. Die Python-Bibliotheken, welche für diesen Schritt verwendet werden, sind nltk, bzw. nltk.tokenize und nltk.corpus.

Damit Daten in numerische Vektoren umgewandelt werden können, kommen die Bag of Words (BoW) und die term frequency times inverse document frequency (TF-IDF) Variante zum Einsatz. Der Output dieser Umwandlung kann anschließend zur Themenfindung genutzt werden. Die jeweiligen Python-Bibliotheken, die für die BoW Technik verwendet werden, sind sklearn CountVectorizer und für TF-IDF TfidfVectorizer.

Die abschließende Themenfindung soll mittels Latent Semantic Analysis (LSA) und Latent Dirichlet Allocation (LDA) realisiert werden. Die hier zu verwenden Python-Bibliotheken sind sklearn TruncatedSVD und LatentDirichletAllocation.
