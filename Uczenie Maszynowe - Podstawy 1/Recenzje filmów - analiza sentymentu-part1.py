""" Preprocessing---
przygotowanie danych tekstowych tak, aby można było zastosować do nich algorytmy uczenia maszynowego.
-- Anazliza danych tekstowych z portalu IMDB;
-- Na podstawie samych słów przewidujemy czy recenzja pozytywna/negatywna
-- Recenzje znajdują się w kolumnie review"""

import pandas as pd
import nltk  # biblioteka do operacji tekstowych
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt

imdb = pd.read_csv('imdb.csv')

# Korpusy językowe - bazy słów

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Tworzymy kolumnę review_processed
imdb['review_processed'] = imdb['review'].str.lower()
imdb['review_processed'] = imdb['review_processed'].str.replace(r'[^a-zA-Z\s]', '', regex=True)
imdb['review_tokens'] = imdb['review_processed'].apply(word_tokenize)

# Lematyzacja i usuwanie stopwords
lemmatizer = WordNetLemmatizer()  # inicjowanie lemmatizera
stop_words = set(stopwords.words('english'))

processed_reviews = []
for review in imdb['review_tokens']:
    processed_review = []
    for word in review:
        if word not in stop_words and word != "br":
            lemmatized_word = lemmatizer.lemmatize(word)
            processed_review.append(lemmatized_word)
    processed_reviews.append(processed_review)

#print(processed_reviews[0])

# Zapisanie wyników lematyzacji

imdb["review_processed"] = [' '.join(review) for review in processed_reviews]
# print(imdb["review_processed"][0])

### EDA

positive_reviews = imdb[imdb['sentiment'] == 'positive']['review_processed']
negative_reviews = imdb[imdb['sentiment'] == 'negative']['review_processed']

all_words_pos = ' '.join(positive_reviews).split()
all_words_neg = ' '.join(negative_reviews).split()

word_freq_pos = Counter(all_words_pos)
word_freq_neg = Counter(all_words_neg)

positive_common_words = word_freq_pos.most_common(20)
negative_common_words = word_freq_neg.most_common(20)

# Wizualizacja
# dwa wykresy typu bar plot ze zliczeniem najczęściej występujących słów dla negatywnych i pozytywnych recenzji

def plot_word_frequency(word_freq, title, ax):
    word, count = zip(*word_freq)
    ax.barh(word, count, color='green')
    ax.set_xlabel('Częstość występowania słów')
    ax.set_title(title)
    ax.invert_yaxis()

fig, axes = plt.subplots(1, 2, figsize=(15,6))

plot_word_frequency(positive_common_words, "20 najczęściej występujących słów - recenzje pozytywne", ax=axes[0])
plot_word_frequency(negative_common_words, "20 najczęściej występujących słów - recenzje negatywne", ax=axes[1])

plt.savefig('most_common_words.png')

# Stwórz kolumnę imdb['review_length'], która reprezentuje liczbę słów w każdej recenzji

imdb['review_length'] = imdb['review_processed'].str.count(r'\w+')