import plotly.express as px
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud, STOPWORDS
from cleaning import lemmatize_text
import os



def remove_custom_stopwords(custom_stopwords):#Not implemented at last#Streamlit issues to add stopwords written by the user
    #custom_stopwords=st.text_input("Add here words to removed from the wordcloud (word1,word2,...)")
    stop_words = list(custom_stopwords) +list(STOPWORDS)
    return stop_words

def generate_individual_wordcloud(word_frequencies, title, ax):
    wordcloud = WordCloud(stopwords=STOPWORDS, width=400, height=400, max_words=20)
    wordcloud.generate_from_frequencies(word_frequencies)

    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(title)
    ax.axis('off')

def generate_histogram(word_frequencies, title, num_words=50):
    words = list(word_frequencies.keys())[:num_words]
    frequencies = list(word_frequencies.values())[:num_words]
    frequencies = [round(freq, 2) for freq in frequencies]

    # Create a DataFrame with the word and frequency data
    df_histogram = pd.DataFrame({'Word': words, 'Normalized Frequency': frequencies})

    # Sort the DataFrame by frequency in descending order
    df_histogram.sort_values(by='Normalized Frequency', ascending=False)
    

    fig = px.bar(x=words, y=frequencies, color_discrete_sequence=['blue'],
                 labels={'x': 'Words', 'y': 'Normalized frequency'})
    fig.update_layout(title=title, xaxis_tickangle=-90)

    return fig

def generate_main_differences_expressions(df_clean, app_id):
    df_clean['cleaned_words'] = df_clean['clean_reviews'].apply(lemmatize_text)
    voted_up = df_clean[df_clean['voted_up'] == True]
    voted_down = df_clean[df_clean['voted_up'] == False]
    categories = [voted_up, voted_down]

    # Create TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(2,3))

    data = []
    for i, category in enumerate(categories):
        text = ' '.join(category['cleaned_words'].astype(str))

        # Fit and transform the text
        tfidf_matrix = tfidf_vectorizer.fit_transform([text])
        # Get the feature names (words)
        feature_names = tfidf_vectorizer.get_feature_names_out()

        # Create a dictionary of word frequencies
        word_frequencies = dict(zip(feature_names, tfidf_matrix.toarray()[0]))

        if i == 0:
            word_frequencies_voted_up = word_frequencies

        # Calculate the difference between voted_up and voted_down scores
        diff_scores = []
        for term, score in word_frequencies.items():
            diff_scores.append(abs(score - word_frequencies_voted_up.get(term, 0)))

        # Create a list of tuples with the word, voted_up score, voted_down score, and difference
        word_data = [
            (term, word_frequencies_voted_up.get(term, 0), score,
             abs(score - word_frequencies_voted_up.get(term, 0)))
            for term, score in word_frequencies.items()]

        data.extend(word_data)

    # Create a DataFrame with the word data
    df_diff_scores = pd.DataFrame(data, columns=['Expression', 'Voted Up Expression Normalized Frequency', 'Voted Down Expression Normalized Frequency Score', 'Score Difference'])

    # Sort the DataFrame by score difference in descending order
    df_diff_scores = df_diff_scores.sort_values(by='Score Difference', ascending=False)

    #create a .csv, put it in semantics
    path2 = './semantics_output/'
    if not os.path.exists(path2):
        os.makedirs(path2)
    # Chemin complet du fichier de sortie
    semantics_file_path = os.path.join(path2, "Expressions_frequencies_voted_up_down_" + app_id + ".csv")
    df_diff_scores.to_csv(semantics_file_path, index=False, sep=";")

def generate_main_differences_words(df_clean, app_id):
    df_clean['cleaned_words'] = df_clean['clean_reviews'].apply(lemmatize_text)
    voted_up = df_clean[df_clean['voted_up'] == True]
    voted_down = df_clean[df_clean['voted_up'] == False]
    categories = [voted_up, voted_down]

    # Create TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1,1))

    data = []
    for i, category in enumerate(categories):
        text = ' '.join(category['cleaned_words'].astype(str))

        # Fit and transform the text
        tfidf_matrix = tfidf_vectorizer.fit_transform([text])
        # Get the feature names (words)
        feature_names = tfidf_vectorizer.get_feature_names_out()

        # Create a dictionary of word frequencies
        word_frequencies = dict(zip(feature_names, tfidf_matrix.toarray()[0]))

        if i == 0:
            word_frequencies_voted_up = word_frequencies

        # Calculate the difference between voted_up and voted_down scores
        diff_scores = []
        for term, score in word_frequencies.items():
            diff_scores.append(abs(score - word_frequencies_voted_up.get(term, 0)))

        # Create a list of tuples with the word, voted_up score, voted_down score, and difference
        word_data = [
            (term, word_frequencies_voted_up.get(term, 0), score,
             abs(score - word_frequencies_voted_up.get(term, 0)))
            for term, score in word_frequencies.items()]

        data.extend(word_data)

    # Create a DataFrame with the word data
    df_diff_scores = pd.DataFrame(data, columns=['Word', 'Voted Up Word Frequency Score', 'Voted Down Word Frequency Score', 'Score Difference'])

    # Sort the DataFrame by score difference in descending order

    # create a .csv, put it in semantics
    path2 = './semantics_output/'
    if not os.path.exists(path2):
        os.makedirs(path2)
    # Chemin complet du fichier de sortie
    df_diff_scores = df_diff_scores.sort_values(by='Score Difference', ascending=False)
    semantics_file_path = os.path.join(path2, "Words_frequencies_voted_up_down_" + app_id + ".csv")
    df_diff_scores.to_csv(semantics_file_path, index=False, sep=";")

