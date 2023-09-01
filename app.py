# basic libraries
import json
import pandas as pd
from scipy import stats
import os
import numpy as np
# load data
import \
    steamreviews_verify_false  # steamreviews_verify_false.py Class pour modifier par héritage la fonction download_reviews_for_app_id_with_offset sans vérification SSL

# visualisation
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

# load (up/down) data and clean data
import re
import base64
from describe_reviews import \
    aggregate_reviews_to_pandas  # Raw data exploration, json into dataframe with sentiment, subjectivity and readibility scores of top
from cleaning import run_data_cleaning, lemmatize_text, \
    run_data_cleaning_universal  # the last method added to allow data preparation of reviews from another source than Steam

# semantics analysis#clustering
from semantic import generate_individual_wordcloud, generate_histogram, generate_main_differences_expressions, \
    generate_main_differences_words  # Dataframe cleaning and filtering - number of languages to keep : a parameter to be customized
from clustersim import text_clustering, word_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# lda_topic_modeling #Sentiment analysis
from lda_topic_modeling import spacy_lemma, make_trigrams, get_sentiment_score, classify_sentiments
import gensim
from gensim.models import CoherenceModel, LdaMulticore
import pyLDAvis.gensim as p_gensim
import pyLDAvis
import streamlit.components.v1 as components


# Author : Chrys Grosso
# Date : 2023-08
# Main sources :
# Woctumeza/Wox, creator of the steamreviews python package https://pypi.org/project/steamreviews/
# Alfred Tang - Latent Topics Modeling for Steam User reviews https: // medium.com / @ alfredtangsw / steamvox - sujet - modélisation - sentiment - analyse - d83a88d3003a"


# 1. PAGE LOAD DATA - EXPLORATION OF RAW DATA : lineS 39 to 100
# 2. PAGE SEMANTICS ANALYSIS (WITH TEXT CLUSTERING AND WORD EMBEDDING): lines 477 to 639
# 3. PAGE LATENT TOPICS MODELING WITH SENTIMENT ANALYSIS: lines 640 to 836
# DEF MAIN(): lines 837 to 1126  - main function to built the frameworks and run the app #when in the terminal, type streamlit run app.py

# 1. PAGE LOAD DATA AND EXPLORATION OF RAW DATA
def load_game_data():
    # Charger le fichier db_table.csv dans un DataFrame
    db_table = pd.read_csv("db/db_table.csv")

    # Créer un dictionnaire associant les appids aux noms de jeux à partir du DataFrame
    games_data = dict(zip(db_table['gameid'].astype(str), db_table['name'].astype(str)))
    return games_data


def search_game_by_name(user_input, games_data):
    # Convertir l'entrée utilisateur en minuscules pour une recherche insensible à la casse
    user_input_lower = user_input.lower()

    # Filtrer les jeux dont le nom commence par les premières lettres saisies par l'utilisateur
    matches = {appid: name for appid, name in games_data.items() if name.lower().startswith(user_input_lower)}
    return matches


def load_data(app_id):  # app_id : Steam ID of the game#Park Beyond : 1368130
    # Data folder
    data_path = "data/"

    if app_id is not None:
        json_filename = "review_" + app_id + ".json"
        data_filename = os.path.join(data_path, json_filename)

        if not os.path.exists(data_filename):
            st.warning("The file does not exist. Downloading in progress... \
            For large files, the process can take up to 5 minutes")
            # Utilisation de db_table_path dans match_app_id_with_name
            with st.empty():
                steamreviews_verify_false.download_reviews_for_app_id(app_id)  # Function to download the data
                # ckeck out main.py to add conditions (recent games, ...)

        else:
            with st.empty():
                st.success("The file exists. Downloading in progress...For a better experience, please wait for the end of the process. \
                For large files, the process can take up to 5 minutes")

        # Charger les données JSON dans un DataFrame Pandas
        with open(data_filename, encoding="utf8") as in_json_file:
            review_data = json.load(in_json_file)
            # Afficher le début du fichier JSON
            # Obtenir le nombre attendu de critiques
            num_reviews = None
            if "query_summary" in review_data:
                query_summary = review_data["query_summary"]
                if "total_reviews" in query_summary:
                    num_reviews = query_summary["total_reviews"]

            # Afficher le nombre attendu de critiques #("[appID = {}] expected reviews = {}".format(app_id, num_reviews))
            # afficher l'appID
            if num_reviews is not None:
                st.write("[appID = {}]".format(app_id))

            else:
                st.warning("Unable to determine the expected number of reviews.")

        rev = aggregate_reviews_to_pandas(app_id)

        return review_data, rev
    else:
        st.error("Invalid app_id or game title.")
        return None, None


def describe_data(review_data):
    query_summary = review_data.get('query_summary')

    if query_summary is not None:
        total_reviews = query_summary.get('total_reviews', 'N/A')
        total_positive = query_summary.get('total_positive', 'N/A')
        total_negative = query_summary.get('total_negative', 'N/A')

        sentence = 'Number of reviews: {0} ({1} voted_up "👍" ; {2} voted_down "👎")'.format(
            total_reviews, total_positive, total_negative
        )

        # Reste du code utilisant les valeurs obtenues

        # Pie chart
        labels = ['Voted_up', 'Voted_down']
        values = [total_positive, total_negative]
        colors = ['green', 'red']
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, marker=dict(colors=colors))])
        fig.update_layout(
            title="Split of voted_up/voted_down",
            font=dict(size=12)
        )
        fig.update_traces(textfont=dict(size=14))  # text size in chart sections
        # Round the percentage values to 2 decimal places in the hover text
        st.plotly_chart(fig)
    else:
        st.write('Query summary cannot be found in the JSON file.')

    reviews = list(review_data.get('reviews', {}).values())
    sentence = 'Number of downloaded reviews: ' + str(len(reviews))
    st.write(sentence)

    return query_summary, review_data


@st.cache_data
def describe_language(review_data):
    if review_data is None or 'reviews' not in review_data:
        st.write("Error loading data.")
        return
        # Graphique interactif de la répartition des avis par langue
    language_counts = {}
    reviews = review_data['reviews']
    for review_id, review in reviews.items():
        language = review.get('language')
        if language in language_counts:
            language_counts[language] += 1
        else:
            language_counts[language] = 1
    # Trier les langues par ordre décroissant des nombres de reviews
    counts, languages = zip(*sorted(zip(language_counts.values(), language_counts.keys()), reverse=True))

    fig = go.Figure(data=[go.Bar(x=languages, y=counts)])
    fig.update_layout(
        title="Breakdown of reviews by language",
        xaxis_title="Language",
        yaxis_title="Number of reviews",
        font=dict(size=12)
    )
    st.plotly_chart(fig)

    voted_up_counts = []
    voted_down_counts = []
    for language in languages:
        voted_up = sum(review['voted_up'] == True for review in reviews.values() if review.get('language') == language)
        voted_down = sum(
            review['voted_up'] == False for review in reviews.values() if review.get('language') == language)
        voted_up_counts.append(voted_up)
        voted_down_counts.append(voted_down)

    if len(languages) > 0:  # Check if languages list is not empty
        fig2 = go.Figure(data=[
            go.Bar(x=languages, y=voted_up_counts, name='Voted Up', marker_color='green'),
            go.Bar(x=languages, y=voted_down_counts, name='Voted Down', marker_color='red')
        ])

        fig2.update_layout(
            title="Breakdown of reviews by language and votes",
            xaxis_title="Language",
            yaxis_title="Number of reviews",
            font=dict(size=12),
            barmode='stack'  # Empile les barres pour les votes up et down
        )

        st.plotly_chart(fig2)
    else:
        st.write("No data available for languages.")

    return review_data


@st.cache_data
def plot_voted_up_down(app_id):
    rev = aggregate_reviews_to_pandas(app_id)

    # 1st graph to visualize reviews per day and year (all/ voted_up/voted_down)
    rev['review_date'] = pd.to_datetime(rev['review_date'], format="%d/%m/%Y %H:%M:%S", errors='coerce')
    # Créer une colonne 'week' pour regrouper par semaine
    rev['week'] = rev['review_date'].dt.to_period('W').astype(str)
    # Calculer le nombre total de revues par jour
    total_reviews_counts = rev.groupby(rev['review_date'].dt.date).size().reset_index(name='total')
    # Calculer le nombre de revues voted_up par jour
    reviews_voted_up_counts = rev[rev['voted_up'] == True].groupby(
        rev['review_date'].dt.date).size().reset_index(name='voted_up')
    # Calculer le nombre de revues voted_down par jour
    reviews_voted_down_counts = rev[rev['voted_up'] == False].groupby(
        rev['review_date'].dt.date).size().reset_index(name='voted_down')

    # create a new directory file for dataframes of the reviews in .csv (will be the same for cleaned reviews)
    path1 = './output/'
    if not os.path.exists(path1):
        os.makedirs(path1)

    output_file_path = os.path.join(path1, "Raw_reviews_" + app_id + ".csv")

    rev.to_csv(output_file_path, index=False, sep=";")  # Save the cleaned dataframe of reviews into .csv

    # Créer le graphique avec plotly express

    # Calculate the percentage of positive reviews (voted_up / total) per day
    reviews_voted_up_counts['percentage_positive'] = reviews_voted_up_counts['voted_up'] / (
                reviews_voted_up_counts['voted_up'] + reviews_voted_down_counts['voted_down']) * 100

    # Create the first graph with plotly express
    fig1 = px.line()

    # Add the line "total"
    fig1.add_scatter(x=total_reviews_counts['review_date'], y=total_reviews_counts['total'], name='Total',
                     line_color='blue')

    # Add the line "voted_up"
    fig1.add_scatter(x=reviews_voted_up_counts['review_date'], y=reviews_voted_up_counts['voted_up'], name='Voted Up',
                     line_color='green')

    # Add the line "voted_down"
    fig1.add_scatter(x=reviews_voted_down_counts['review_date'], y=reviews_voted_down_counts['voted_down'],
                     name='Voted Down', line_color='red')

    # Update the parameters of the first graph
    fig1.update_layout(
        title='Number of Reviews per day and year',
        xaxis_title='Date',
        yaxis_title='Number of Reviews',
        hovermode='x',
        legend=dict(x=0.5, y=1.15, orientation='h', bgcolor='rgba(255, 255, 255, 0)',
                    bordercolor='rgba(255, 255, 255, 0)')
    )

    # Create the second graph (bar chart) with plotly express
    fig2 = px.bar(reviews_voted_up_counts, x='review_date', y='percentage_positive',
                  title='Percentage of Positive Reviews per day and year')

    # Update the parameters of the second graph
    fig2.update_layout(
        xaxis_title='Date',
        yaxis_title='Percentage of Positive Reviews',
        hovermode='x',
        showlegend=False
    )
    # Set the bar color to green
    fig2.update_traces(marker=dict(color='green'))

    # Round the percentage values to 2 decimal places in the hover text
    fig2.update_traces(hovertemplate='%{y:.2f}%')

    # Combine both graphs side by side
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)

    # 2nd graph to visualize reviews per payment_status and voted_up/voted_down
    # Filter data for paid and received for free
    paid_game = rev[rev['received_for_free'] == False]
    received_for_free = rev[rev['received_for_free'] == True]

    # Calculate counts for paid and received for free
    paid_count = len(paid_game)
    received_for_free_count = len(received_for_free)

    # Filter data for voted_up and voted_down
    voted_up_paid = paid_game[paid_game['voted_up']]
    voted_up_received = received_for_free[received_for_free['voted_up']]
    voted_down_paid = paid_game[~paid_game['voted_up']]
    voted_down_received = received_for_free[~received_for_free['voted_up']]

    # Calculate counts for voted_up and voted_down
    voted_up_paid_count = len(voted_up_paid)
    voted_up_received_count = len(voted_up_received)
    voted_down_paid_count = len(voted_down_paid)
    voted_down_received_count = len(voted_down_received)

    # Create the bar chart
    fig2 = go.Figure(data=[
        go.Bar(x=['Overall', 'Voted Up', 'Voted Down'], y=[paid_count, voted_up_paid_count, voted_down_paid_count],
               name='Paid'),
        go.Bar(x=['Overall', 'Voted Up', 'Voted Down'],
               y=[received_for_free_count, voted_up_received_count, voted_down_received_count],
               name='Received for Free')
    ])

    fig2.update_layout(
        title="Breakdown of Paid/Received for Free and Voted Up/Voted Down",
        xaxis_title="Category",
        yaxis_title="Count",
        font=dict(size=12),
        barmode='group'
    )

    st.plotly_chart(fig2, use_container_width=True)
    st.write("It might be possible that paying players have more reasons to give frank reviews.")

    # Graph3 - Dectection of spammers and duplicates
    steamid_count = rev['steamid'].nunique()
    reviews_count = rev['review'].nunique()

    categories = ['steamid', 'review']
    counts = [steamid_count, reviews_count]

    fig3 = go.Figure(data=[go.Bar(x=categories, y=counts)])

    fig3.update_layout(
        title="Spammers / Duplicates Detection",
        xaxis_title="Categories",
        yaxis_title="Number of Reviews",
        font=dict(size=12)
    )
    st.plotly_chart(fig3)
    st.write(
        "If the number of unique reviews is equal to the number of steamid, that means that there are no duplicates.")
    st.write(" If the number of unique reviews is lower than steamids, that means that some reviews have been spammed several times and should be removed\
     to conduct the review analysis.")

    # 4th graph : Average Review length
    mean_length_voted_up = round(rev.loc[rev['voted_up'], 'character_count'].mean(), 2)
    mean_length_voted_down = round(rev.loc[~rev['voted_up'], 'character_count'].mean(), 2)

    # Création du graphique à barres
    labels = ['Voted Up', 'Voted Down']
    colors = ['green', 'red']
    mean_lengths = [mean_length_voted_up, mean_length_voted_down]

    fig4 = go.Figure(data=[go.Bar(x=labels, y=mean_lengths, marker=dict(color=colors))])

    fig4.update_layout(
        title="Average Review Length by Vote ",
        xaxis_title="Vote",
        yaxis_title="Average Number of Characters",
        font=dict(size=12)
    )

    voted_up_lengths = rev.loc[rev['voted_up'], 'character_count']
    voted_down_lengths = rev.loc[~rev['voted_up'], 'character_count']

    t_statistic, p_value = stats.ttest_ind(voted_up_lengths, voted_down_lengths)

    if p_value < 0.05:
        result = "statistically significant"
    else:
        result = "not statistically significant"

    # Afficher le graphique
    st.plotly_chart(fig4)
    st.write(f"Test t: t-statistic = {t_statistic:.2f}, p-value = {p_value:.4f}")
    st.write(f"The difference in review lengths is {result} at a 95% confidence level.")

    # 5th graph: Box plot
    voted_up = rev.loc[rev['voted_up'], 'character_count']
    voted_down = rev.loc[~rev['voted_up'], 'character_count']

    # Création du graphique en box plot
    data = [
        go.Box(y=voted_up, name='Voted Up', boxpoints='outliers', marker_color='green'),
        go.Box(y=voted_down, name='Voted Down', boxpoints='outliers', marker_color='red')
    ]

    layout = go.Layout(
        title="Review Length by Vote",
        xaxis_title="Vote",
        yaxis_title="Review Length",
        font=dict(size=12)
    )

    fig5 = go.Figure(data=data, layout=layout)
    st.plotly_chart(fig5)
    st.write(
        f"For the rest of the analysis, we shall remove outlier reviews with lengths less than 10 characters or greater than 4000 characters (nest of spams like 8k characters of m in a row, cooking recipes - or of pseudo-expert opinions).")
    # 6th graph : Box plot
    # Remove outlier reviews with length less than 10 or greater than 4000
    voted_up = voted_up[(voted_up >= 10) & (voted_up <= 4000)]
    voted_down = voted_down[(voted_down >= 10) & (voted_down <= 4000)]

    # Création du graphique en box plot
    data = [
        go.Box(y=voted_up, name='Voted Up', boxpoints='outliers', marker_color='green'),
        go.Box(y=voted_down, name='Voted Down', boxpoints='outliers', marker_color='red')
    ]

    layout = go.Layout(
        title="Review Length without outliers",
        xaxis_title="Vote",
        yaxis_title="Review Length",
        font=dict(size=12)
    )

    fig6 = go.Figure(data=data, layout=layout)
    st.plotly_chart(fig6)

    st.write(f"Last check : to be credible, the author should have played at least 3 hours on Steam.")
    # 7th graph -Playtime forever and last two weeks
    # Création du graphique en box plot
    playtime_forever = rev['playtime_forever'] / 60
    playtime_two_weeks = rev['playtime_last_two_weeks'] / 60
    # Round the values to 2 decimal places
    playtime_forever = np.round(playtime_forever, 2)
    playtime_two_weeks = np.round(playtime_two_weeks, 2)
    data7 = [
        go.Box(y=playtime_forever, name='Playtime forever', boxpoints='outliers'),
        go.Box(y=playtime_two_weeks, name='Playtime last two weeks', boxpoints='outliers')
    ]

    layout7 = go.Layout(
        title="Playtime forever and last two weeks",
        xaxis_title="Playtime ",
        yaxis_title="Hours",
        font=dict(size=12)
    )

    fig7 = go.Figure(data=data7, layout=layout7)
    # Format the hover text to display hours with 2 decimal places
    st.plotly_chart(fig7)

    path1 = './output/'

    st.write(
        "For the purpose of automated review analysis, we should also exclusively retain reviews written in English, which represents the top language category.")
    df_clean = run_data_cleaning(rev).reset_index(drop=True)
    # path
    output_file_path = os.path.join(path1, "Filtered_reviews_" + app_id + ".csv")

    df_clean.to_csv(output_file_path, index=False, sep=";")  # Save the cleaned dataframe of reviews into .csv

    deduplicated_rows = abs(len(rev) - len(df_clean))
    st.write("Dataframe length after cleaning and focussing on reviews written in english : ", len(df_clean),
             "Rows removed :", deduplicated_rows, round(deduplicated_rows / len(rev) * 100, 2), "%")

    st.write(df_clean)

    return rev, df_clean

def download(app_id):
    output_path = './output/'

    # Add a download button (filtered/cleaned file)
    filtered_file_path = os.path.join(output_path, "Filtered_reviews_" + app_id + ".csv")
    with open(filtered_file_path, "rb") as file:
        st.download_button(
            label="Download Filtered_reviews",
            data=file,
            file_name="Filtered_reviews_" + app_id + ".csv",
            mime="text/csv"
        )

    # Add a download button (raw file)
    raw_file_path = os.path.join(output_path, "Raw_Reviews_" + app_id + ".csv")
    with open(raw_file_path, "rb") as file:
        st.download_button(
            label="Download Raw data",
            data=file,
            file_name="Raw_Reviews_" + app_id + ".csv",
            mime="text/csv"
        )


# PAGE 2 -SEMANTICS ANALYSIS

def generate_unigram_wordcloud3(df_clean, app_id):
    df_clean['cleaned_words'] = df_clean['clean_reviews'].apply(lemmatize_text)
    voted_up = df_clean[df_clean['voted_up'] == True]
    voted_down = df_clean[df_clean['voted_up'] == False]
    categories = [voted_down, voted_up]
    titles = ['Voted Down', 'Voted Up']

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # wordclouds voted_up, voted_down
    for i, category in enumerate(categories):
        text = ' '.join(category['cleaned_words'].astype(str))

        # Create TF-IDF vectorizer
        tfidf_vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(2, 3))

        # Fit and transform the text
        tfidf_matrix = tfidf_vectorizer.fit_transform([text])

        # Get the feature names (words)
        feature_names = tfidf_vectorizer.get_feature_names_out()

        # Create a dictionary of word frequencies
        exp_frequencies = dict(zip(feature_names, tfidf_matrix.toarray()[0]))

        # Remove bigramms composed of the same words
        for bigram_trigram in list(exp_frequencies.keys()):
            exprs = re.findall(r'\b\w+\b', bigram_trigram)
            if len(exprs) == 2 and exprs[0] == exprs[1]:
                exp_frequencies.pop(bigram_trigram)

        # Sort the word frequencies in descending order
        # updated_word_frequencies = [(word, frequency) for word, frequency in word_frequencies.items()]
        sorted_exp_frequencies = sorted(exp_frequencies.items(), key=lambda x: x[1], reverse=True)

        # Calculate the number of terms to be removed
        num_terms = len(sorted_exp_frequencies)
        num_terms_to_remove = int(0.001 * num_terms)  # 0,1% of the total terms

        # Separate the top word frequencies based on TF-IDF scores
        top_exp_frequencies = dict(sorted_exp_frequencies[num_terms_to_remove:])

        ax = axes[i]  # Set the subplot
        generate_individual_wordcloud(top_exp_frequencies, titles[i], ax)

    # fig2 = plt.subplots(figsize=(12, 6))

    # histogram and df all_reviews voted_up+voted_down#Combine the text from all categories
    text = ' '.join(df_clean['cleaned_words'].astype(str))

    # Create TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 3))

    # Fit and transform the text
    tfidf_matrix = tfidf_vectorizer.fit_transform([text])

    # Get the feature names (words)
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # Create a dictionary of word frequencies
    word_frequencies = dict(zip(feature_names, tfidf_matrix.toarray()[0]))

    # Remove bigramms composed of the same words
    for bigram_trigram in list(word_frequencies.keys()):
        words = re.findall(r'\b\w+\b', bigram_trigram)
        if len(words) == 2 and words[0] == words[1]:
            word_frequencies.pop(bigram_trigram)

    # Sort the word frequencies in descending order
    sorted_word_frequencies = sorted(word_frequencies.items(), key=lambda x: x[1], reverse=True)

    # Create a DataFrame with the sorted word frequencies
    df_histogram = pd.DataFrame(sorted_word_frequencies, columns=['Expression/Word', 'Normalized Frequency'])

    # Save the DataFrame to a CSV file in semantics_output file
    path2 = './semantics_output/'
    # path
    output_file_path = os.path.join(path2, "Expressions_and_words_frequencies_all_reviews_" + app_id + ".csv")

    df_histogram.to_csv(output_file_path, index=False, sep=";")  # Sauvegarder en csv

    # Separate the top word and expressions frequencies based on TF-IDF scores
    top_word_normalized = dict(sorted_word_frequencies)

    fig2 = generate_histogram(top_word_normalized, 'Normalized Frequencies of words and expressions All reviews')

    plt.tight_layout()
    st.session_state = top_word_normalized
    st.pyplot(fig)
    st.plotly_chart(fig2)
    return df_clean


# Function to generate download link
def get_download_link(file, file_name, text):
    file_content = file.read()
    b64 = base64.b64encode(file_content).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{file_name}">{text}</a>'
    return href


# Function to generate the plot of similarity
def generate_similarity_plot(df_clean, n_wordy, app_id):
    st.subheader('Automated similarity plot')
    D, app_id = word_similarity(df_clean, app_id)  # D is a DataFrame with the similarity between words

    # Save the DataFrame to a CSV file in vectors_output file
    path2b = './word2vec_vectors/'
    # path
    output_file_path = os.path.join(path2b, "words.vectors_" + app_id + ".csv")
    D.to_csv(output_file_path, sep=";", header=True)  # Sauvegarder en csv

    # Add a download button (vectors created with word2vec)
    vectors_path = os.path.join(path2b, "words.vectors_" + app_id + ".csv")
    with open(vectors_path, "rb") as file:
        st.markdown(get_download_link(file, "words.vectors_" + app_id + ".csv",
                                      "Download words vectors for further analysis"),
                    unsafe_allow_html=True)

    # Add a download button (vectors created with word2vec)

    top_words = D.index[:n_wordy].tolist()

    # Créer le graphique de scatter avec hover_name
    fig = px.scatter(D.head(n_wordy), x='V1', y='V2', hover_data=[top_words], hover_name=top_words,
                     labels={'index': 'Mot'})

    # Créer une liste pour les annotations
    annotations = []

    # Ajouter les annotations
    for i, word in enumerate(top_words):
        x_val = D.loc[word, 'V1']
        y_val = D.loc[word, 'V2']
        annotations.append(
            go.Scatter(
                x=[x_val],
                y=[y_val],
                mode='text',
                text=[word],
                textposition='bottom right',
                name=word,  # Nom spécifique pour chaque annotation#f'Term {i+1}'
                textfont=dict(size=14),  # Taille de police personnalisée
            )
        )

    # Ajouter toutes les annotations d'un coup à la figure
    for annotation in annotations:
        fig.add_trace(annotation)

    # Ajuster la taille du graphique
    fig.update_layout(width=900, height=900)

    # Afficher le graphique
    st.plotly_chart(fig, use_container_width=True)
    return D


def generate_custom_sim_plot(D):
    st.subheader('Customized similarity plot')
    st.write('You can enter the words for which you want to see their semantic relationship with other words.')
    st.write('The words must be actualy present in the reviews.')
    # Get user input for the list of words
    user_input = st.text_input("Enter the words separated by commas :")

    # Check if the user has entered any words
    if user_input.strip():
        # Convert the user input to a list of words
        mots = user_input.split(', ')

        # Display the list of words
        st.write("List of words entered by the user:")
        st.write(mots)
        dfMots = D.loc[mots, :]
        fig = px.scatter(dfMots, x='V1', y='V2', hover_data=[dfMots.index], hover_name=dfMots.index,
                         labels={'index': 'Mot'})

        # Ajouter les annotations pour les 10 premiers termes
        for word in dfMots.index:
            x_val = dfMots.loc[word, 'V1']
            y_val = dfMots.loc[word, 'V2']
            fig.add_trace(
                go.Scatter(
                    x=[x_val],
                    y=[y_val],
                    mode='text',
                    text=[word],
                    textposition='bottom right',
                )
            )

        # Ajuster la taille du graphique
        fig.update_layout(width=600, height=800)

        # Afficher le graphique
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Show an info message if the input is empty
        st.info("Enter the words to generate the graph.")


# PAGE 3 - LATENT TOPICS MODELING (LDA) WITH SENTIMENT ANALYSIS

def prepare_lda(df_clean):
    df_clean['3gram_reviews'] = make_trigrams(df_clean)
    df_clean['3grams_nouns_verbs_adj'] = df_clean['3gram_reviews'].map(
        lambda x: spacy_lemma(x, allowed_postags=['NOUN', 'VERB', 'ADJ']))

    # Save the DataFrame to a CSV file in n_grams_lda file
    path3 = './n_grams_lda/'
    # path
    grams_file_path = os.path.join(path3, "final_df_grams.csv")

    df_clean.to_csv(grams_file_path, index=False, sep=";")  # csv

    return df_clean


def run_topic_modeling(df_clean):
    # Fine-tuning of the model and adaptation for Streamlit
    # https: // medium.com / @ alfredtangsw / steamvox - topic - modelling - sentiment - analysis - d83a88d3003a
    # Build dictionary and corpus from 3gram dataset, NOUNS VERBS ADJECTIVES with filter_extremes()
    # documents = df_clean['3grams_nouns_verbs_adj']
    documents = [eval(doc) for doc in df_clean['3grams_nouns_verbs_adj']]
    dictionary = gensim.corpora.Dictionary(documents)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus = [dictionary.doc2bow(word) for word in documents]

    # LDA model parameters
    num_topics = 4
    passes = 100
    eval_every = None
    ldamodel1 = LdaMulticore(corpus, num_topics=num_topics, id2word=dictionary, passes=passes, alpha=2,
                             eval_every=eval_every, workers=3, random_state=42)

    # Check resulting topics
    topic_list = ldamodel1.print_topics(num_topics=num_topics, num_words=15)
    st.markdown("## Check resulting topics")
    for index, i in enumerate(topic_list):
        str1 = str(i[1])
        str1 = str1.replace("'", "").replace("[", "").replace("]", "").replace(" ", "")

        for c in "0123456789+*\".":
            str1 = str1.replace(c, " ")
        # st.write(f"The topic {index+1} can be described as:")
        st.write("One of the 4 topics can be described as:")
        st.write(str1)

    # Compute Perplexity
    st.markdown("## Model quality")
    st.markdown(
        "Perplexity is a measure of how good the model is, of its ability to predict new samples. The lower, the better. A good score starts from around -6.")
    # a measure of how good the model is. lower the better.
    st.write('\nPerplexity:', round(ldamodel1.log_perplexity(corpus),2))

    # Compute Coherence Score
    coherence_model_lda1 = CoherenceModel(model=ldamodel1, texts=documents, dictionary=dictionary, coherence='c_v')
    coherence_lda1 = coherence_model_lda1.get_coherence()
    st.markdown("Coherence score is a measure that assesses the quality and interpretability of topics. \
    A good coherence score typically falls within a range of 40% to 65 /70%. For small datasets (less than 500 reviews), \
    a coherence score around 40% can be considered reasonably good. 65%/70% can be achieved with samples smaller than 10k reviews.\
    Due to the probabilistic nature of the model and the unsupervised learning process, \
    the model's output may vary slightly accross different runs."
                )
    st.write('\nCoherence Score:', round(coherence_lda1, 2))

    # Generate pyLDAvis visualization
    # https: // nbviewer.org / github / bmabey / pyLDAvis / blob / master / notebooks / pyLDAvis_overview.ipynb
    vis = p_gensim.prepare(ldamodel1, corpus, dictionary)

    # Save the model
    newpath = './topic_modeling/'
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    ldamodel1.save('./topic_modeling/trigrams_4_topics.model')

    return vis, ldamodel1, corpus, documents


def format_topics_sentences(final_df_grams, ldamodel1, corpus, documents):
    topic_dict = {'0': 'Topic 1',
                  '1': 'Topic 2',
                  '2': 'Topic 3',
                  '3': 'Topic 4'}

    # Init output
    rows = []
    # Get main topic in each document
    for i, row in enumerate(ldamodel1[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)

        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):

            if j == 0:  # => dominant topic
                wp = ldamodel1.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                row_data = [topic_dict[str(topic_num)], round(prop_topic, 2), topic_keywords]
                row_data.extend([documents[i], final_df_grams['review'].iloc[i]])

                rows.append(row_data)
                break

    sent_topics_df = pd.DataFrame(rows, columns=['dominant_topic', 'topic_perc_contribution', 'topic_Keywords',
                                                 'clean_review', 'original_review'])

    return sent_topics_df, topic_dict


def calculate_topic_sentiments(df_dominant_topics, topic_dict):
    perc_dict = {}
    topic_dataframes = []  # List to store DataFrames for each topic

    for topic in list(topic_dict.values()):
        topic_mask = df_dominant_topics['dominant_topic'] == topic
        topic_df = df_dominant_topics[topic_mask]
        sentiment_lst = topic_df['polarity_score']

        topic_pos = [x for x in sentiment_lst if x > 0.1]
        topic_neg = [x for x in sentiment_lst if x < 0]
        topic_neutral = [x for x in sentiment_lst if x >= 0 and x <= 0.1]
        topic_total = len(sentiment_lst)

        topic_pos_perc = len(topic_pos) / topic_total
        topic_neutral_perc = len(topic_neutral) / topic_total
        topic_neg_perc = len(topic_neg) / topic_total

        perc_dict[topic] = [round(topic_pos_perc, 2), round(topic_neutral_perc, 2), round(topic_neg_perc, 2)]

        topic_dataframes.append(topic_df)  # Add the DataFrame for the current topic to the list

    # Concatenate DataFrames for all topics
    all_topic_df = pd.concat(topic_dataframes, ignore_index=True)
    # Save the Dataframe
    path = './topic_modeling/'
    # path
    topic_sent_path = os.path.join(path, "topics_sentiment.csv")

    all_topic_df.to_csv(topic_sent_path, index=False, sep=";")  # csv

    return perc_dict, topic_dict


def create_sentiment_chart(perc_dict, global_sentiments):
    #
    topic_order = ['Topic 1', 'Topic 2', 'Topic 3',
                   'Topic 4'][::-1]  # reverse the order

    # Get the sentiment percentages for each topic in the desired order
    pos_perc = [perc_dict[topic][0] for topic in topic_order]
    neutral_perc = [perc_dict[topic][1] for topic in topic_order]
    neg_perc = [perc_dict[topic][2] for topic in topic_order]

    fig = go.Figure()
    fig.add_trace(go.Bar(name='Negative', y=topic_order, x=neg_perc, orientation='h', marker_color='red',
                         text=[f"{perc:.1%}" for perc in neg_perc], textposition='inside'))
    fig.add_trace(go.Bar(name='Neutral', y=topic_order, x=neutral_perc, orientation='h', marker_color='gray',
                         text=[f"{perc:.1%}" for perc in neutral_perc], textposition='inside'))
    fig.add_trace(go.Bar(name='Positive', y=topic_order, x=pos_perc, orientation='h', marker_color='green',
                         text=[f"{perc:.1%}" for perc in pos_perc], textposition='inside'))

    fig.update_traces(textfont_size=14)

    fig.update_layout(
        yaxis_title='Game Feature',
        barmode='group',
        xaxis_tickformat='%{0:.0%}',  # Set x-axis tick format to percentage without decimal places
        showlegend=False,
        legend=dict(
            x=0.75,
            y=1.1,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)'
        )
    )
    fig.update_xaxes(tickvals=[])

    # Créer le pie chart des sentiments globaux
    labels = ['Positive', 'Neutral', 'Negative']
    values = [global_sentiments[0], global_sentiments[1], global_sentiments[2]]
    colors = ['green', 'gray', 'red']

    fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, marker=dict(colors=colors))])

    # Afficher le titre principal ("Sentiment Analysis") en haut
    st.title("Sentiment Analysis")
    st.subheader('Sentiment Polarity Overall')

    # Afficher le pie chart des sentiments globaux
    st.plotly_chart(fig_pie, use_container_width=True)

    # Afficher le graphique de barre
    st.subheader('Sentiment Polarity by Game Feature')
    st.plotly_chart(fig, use_container_width=True)


def main():
    st.title("Automated Steam User Reviews Analysis - ASURA")

    # Menu sur le côté avec le composant sidebar
    st.sidebar.title("Menu - ASURA")
    choix = st.sidebar.selectbox("Choose a tab",
                                 ["Welcome","Load data", "Semantics analysis", "Latent Topics_Sentiment analysis"])

    if choix == "Welcome":
        st.title("Welcome to the ASURA application!")
        st.write("""
            <style>
                p {
                    text-align: justify;
                }
            </style>
            <p>We invite you to explore user reviews from Steam for the game of your choice through an interactive journey of graphics,\
             and to take away valuable insights about player expectations and hidden market trends.</p>
             """, unsafe_allow_html=True)

        st.subheader("Here is the journey:")
        st.write("PAGE 1 - LOADING AND DISCOVERING DATA :  DISCOVER THE DATASET AND RELEVANT REVIEWS RETAINED")
        st.write("PAGE 2 - SEMANTICS ANALYSIS : IDENTIFY AREAS OF ACTION")
        st.write("- Key subjects of customer satisfaction and dissatisfaction,")
        st.write("- Potential issues to be resolved,")
        st.write("- Reference competitive landscape. ")
        st.write("PAGE 3 - LATENT TOPICS WITH SENTIMENT TONE : ENGAGE IN UNCOVERING SIGNIFICANT HIDDEN TRENDS WITHIN THE DATA")

        st.subheader("Enjoy your insightful discovery!")


    elif choix == "Load data":
        st.title("Loading and discovering data")
        st.markdown("<hr>", unsafe_allow_html=True)
        # st.header("Search for Steam Games by name")

        # Charger les données du jeu
        games_data = load_game_data()

        # Demander à l'utilisateur de saisir le terme de recherche
        user_input = st.text_input(
            "You can type the first few letters of the title to find out its Steam AppID. The more you write, the more accurate the results:")

        if user_input:
            # Rechercher les jeux correspondants
            matches = search_game_by_name(user_input, games_data)

            # Afficher les correspondances (nom et appid)
            if matches:
                st.subheader("Matching games :")
                for appid, name in matches.items():
                    st.write(f"AppID: {appid}, Name: {name}")
            else:
                st.write("No matching game found.")
        app_id = st.text_input(
            "Copy Paste the afferent AppID Steam of the game. Example : for Park Beyond, its Steam AppID is 1368130. ")
        st.button("Load Data")
        st.session_state['app_id'] = app_id
        if app_id:
            review_data, rev = load_data(app_id)  # Charge les données à partir de l'ID Steam
            st.success("Data loaded successfully.")
            describe_data(review_data)  # Affiche les informations générales voted_up/voted_down sur les données
            describe_language(review_data)  # Affiche la répartition des avis par langue
            plot_voted_up_down(app_id)
            download(app_id)
            # Stocker df_clean dans st.session_state
            df_clean = run_data_cleaning(rev).reset_index(drop=True)
            st.session_state['df_clean'] = df_clean
            st.session_state['app_id'] = app_id
            st.success(
                "End of the discovering step.")


    elif choix == "Semantics analysis":
        st.markdown('<script>window.scrollTo(0,0);</script>', unsafe_allow_html=True)
        app_id = st.session_state['app_id']
        st.title("Semantics analysis")
        st.markdown("<hr>", unsafe_allow_html=True)
        if 'df_clean' in st.session_state:
            df_clean = st.session_state['df_clean']
            # df_clean = st.file_uploader("Upload the cleaned reviews csv.file (Filtered_reviews can be downloaded on the main page Load data)",
            # type=["csv"])

            st.write("The first rows of the dataframe with the cleaned reviews:")
            st.dataframe(df_clean.head(5))
            total_word_count = df_clean['clean_reviews'].apply(lambda x: len(str(x).split())).sum()
            st.write('Number of words in reviews:', total_word_count)
            st.title('Wordclouds')
            st.write(
                'Most frequent expressions in reviews - Normalization with tfidf vectorizer - Empty words mostly retrieved:')
            st.markdown("<hr>", unsafe_allow_html=True)
            generate_unigram_wordcloud3(df_clean, app_id)
            st.markdown("<hr>", unsafe_allow_html=True)
            generate_main_differences_expressions(df_clean, app_id)
            generate_main_differences_words(df_clean, app_id)
            st.session_state['df_clean'] = df_clean
            st.session_state['app_id'] = app_id

            # blue hyperlinks to offer the downloading of main differencies regarding words and expressions
            st.write("For futher analysis, you shall download the csv files:")
            output_path = './semantics_output/'
            semantics_file_path = os.path.join(output_path,
                                               "Expressions_and_words_frequencies_all_reviews_" + app_id + ".csv")
            with open(semantics_file_path, "rb") as file:
                st.markdown(get_download_link(file, "Expressions_and_words_frequencies_all_reviews_" + app_id + ".csv",
                                              "Download words and expressions distributions for all reviews"),
                            unsafe_allow_html=True)

            # Add a download button (filtered/cleaned file)
            semantics2_file_path = os.path.join(output_path, "Expressions_frequencies_voted_up_down_" + app_id + ".csv")
            with open(semantics2_file_path, "rb") as file:
                st.markdown(get_download_link(file, "Expressions_frequencies_voted_up_down_" + app_id + ".csv",
                                              "Download main expressions for voted_up versus main expressions for voted_down"),
                            unsafe_allow_html=True)
            semantics3_file_path = os.path.join(output_path, "Words_frequencies_voted_up_down_" + app_id + ".csv")
            with open(semantics3_file_path, "rb") as file:
                st.markdown(get_download_link(file, "Words_frequencies_voted_up_down_" + app_id + ".csv",
                                              "Download main words for voted_up versus main words for voted_down"),
                            unsafe_allow_html=True)

            st.header('Text Clustering')
            option_text_clustering = st.radio("Would you like to visualize the main clusters of words ?", ["No",
                                                                                                      "Yes"])
            st.write("Selected option for Text Clustering:", option_text_clustering)


            if option_text_clustering == "Yes":
                app_id = st.session_state['app_id']
                st.write('Look on the curve where the bend is formed and choose the number of clusters accordingly.')
                n_cluster = st.sidebar.slider('How many Clusters would you like? ', 0, 7, 3)
                n_word = st.sidebar.slider('How many words per cluster would you like? ', 0, 50, 10)
                df_clean.clean_reviews = df_clean.clean_reviews.apply(lemmatize_text)
                df_clean['clean_review'] = df_clean['clean_reviews'].apply(lambda x: ' '.join(x))
                output = text_clustering(df_clean, 'clean_review', n_cluster,
                                         n_word)
                st.write(output[0])
                st.write(output[1])
                # Add title for top words
                st.subheader("Top words per cluster")
                # Show the top words figure
                st.plotly_chart(output[2])

            st.header('Word similarity')
            option_text_similarity = st.radio("Would you like to see word similarity ? \
                    You could visualize semantic relationships, proximity between words in the context of the reviews’ content:",
                            ["No",
                            "Yes"])
                
            st.write("Selected option for Word similarity:", option_text_similarity)

            if option_text_similarity == "Yes":
                app_id = st.session_state['app_id']
                n_wordy = st.sidebar.slider('How many words would you like to see? ', 10, 100, 50)
                df_clean.clean_reviews = df_clean.clean_reviews.apply(lemmatize_text)
                df_clean['clean_review'] = df_clean['clean_reviews'].apply(lambda x: ' '.join(x))
                D = generate_similarity_plot(df_clean, n_wordy, app_id)
                st.session_state['D'] = D
                generate_custom_sim_plot(D)
                st.markdown("<hr>", unsafe_allow_html=True)
            
            st.success("End of this page.")

    elif choix == "Latent Topics_Sentiment analysis":
        st.markdown('<script>window.scrollTo(0,0);</script>', unsafe_allow_html=True)
        app_id = st.session_state['app_id']
        # Introduction
        st.title("Latent topics with Sentiment analysis")
        st.markdown("<hr>", unsafe_allow_html=True)
        st.header("Introduction")

        # Current path
        current_directory = os.path.dirname(os.path.abspath(__file__))

        # Buid absolute path of the "images" file by combining it with the current path
        images_directory = os.path.join(current_directory, "images")

        # Build absolute path of the image "sentiment_analysis_park_beyond.png" by combining it the "images" path
        image_path = os.path.join(images_directory, "mosaic.png")

        # Open image
        image2 = Image.open(image_path)
        # Load image
        st.image(image2, caption="Image Caption", use_column_width=True)
        st.write("Latent topics are hidden subjects that can be discovered within a collection of documents. \
        NLP (Natural Language Processing) algorithms allow us to identify and quantify the significance of these topics, \
        along with their sentiment tone. Specifically, for user reviews of video games, we have uncovered four main topics that correspond to 'Game Features'. \
        With this information, we can now accurately assess and analyze the sentiment associated with these topics for reviews of any video game:")
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            "**1. Content and authenticity. Examples of keywords (not exhaustive): iconic games titles, content, authenticity, theme.**")
        st.markdown(
            "**2. Characters : keywords like 'character', name of the characters or entities. Example : for Park Beyond, we would find the keyword 'Guest'.**")
        st.markdown(
            "**3. Tactical and Strategic Gameplay. Examples of keywords (not exhaustive) : challenge, gameplay, control, atmosphere.**")
        st.markdown(
            "**4. UX/ UI performance, design, functionalities. Examples of keywords (not exhaustive): design, fun, use, bug, add.**")
        # Insérer un lien hypertexte avec st.write et st.markdown
        url = "https://medium.com/@alfredtangsw/steamvox-topic-modelling-sentiment-analysis-d83a88d3003a"
        st.write("Topics defined from a research paper and fine-tuned : [Link](", url, ")")
        # https: // medium.com / @ alfredtangsw / steamvox - topic - modelling - sentiment - analysis - d83a88d3003a
        # The pre-trained model, built upon these topics and multiple player reviews, delivers more reliable results than starting from scratch."

        st.markdown("<hr>", unsafe_allow_html=True)
        st.header("Discover the significance of game features along with their sentiment tone")
        st.write("Data preparation is crucial for optimal model performance. It may take some time, ranging from 2 minutes for small datasets (1000 reviews) \
        to around 1 hour for larger ones (50k reviews). If you've already completed this step or prepared your data independently, \
         feel free to upload them directly [Upload prepared data for Latent Topics Modeling].")
        # If not, click first on Prepare data for Latent Topics modeling and wait for the file."
        # Offer the user the choice between "Continue with the same model" and "Download a new model"

        option_lda = st.radio("Choose an option:", ["Nothing","Continue with the same reviews",
                                                "Load a new dataset to be prepared",
                                                "Upload prepared data for Latent Topics Modeling"])
        st.write("Selected option:", option_lda)  # Ajout du message de débogage

        if option_lda == "Continue with the same reviews":
            if 'df_clean' not in st.session_state:
                st.error("Please load or prepare the review dataset first.")
            else:
                df_loaded = st.session_state['df_clean']
                st.write(df_loaded.head(5))
                with st.spinner("Data preparation in progress"):
                    prepare_lda(df_loaded)  # Function to create 3grams and to postag
                    st.success("Data prepared successfully.")
                    st.write(
                        "You can now download the file then drag & drop it in in [Upload prepared data for Latent Topics Modeling] above.")

                    # blue hyperlinks to offer the downloading of dataframe with 3grams and postags NOUN, VERB, ADJ

                    output_path = './n_grams_lda/'
                    gram_file_path = os.path.join(output_path, "final_df_grams.csv")
                    with open(gram_file_path, "rb") as file:
                        st.markdown(get_download_link(file, "final_df_grams.csv",
                                                      "Download prepared data for Latent Topics Modeling"),
                                    unsafe_allow_html=True)

        elif option_lda == "Load a new dataset to be prepared":
            # Allow the user to upload a new review dataset
            uploaded_file = st.file_uploader("Upload the Review Dataset", type=".csv")
            st.write("You can upload any reviews file you want. \
                   The column of the reviews to be prepared has to be named 'review' (already the case for the 'Filtered_reviews' file available on the LOAD DATA page of app.")
            if uploaded_file is not None:
                df_loaded1 = pd.read_csv(uploaded_file, sep=';')
                st.write(df_loaded1.head(5))
                run_data_cleaning_universal(df_loaded1)
                # df_loaded1['clean_reviews'] = df_loaded1['clean_reviews'].apply(lemmatize_text)
                with st.spinner("Data preparation in progress"):
                    prepare_lda(df_loaded1)  # Function to create 3grams and to postag

                st.success("Data prepared successfully.")
                st.write(
                    "You can now download the file then drag & drop it in [Upload prepared data for Latent Topics Modeling] above.")

                # blue hyperlinks to offer the downloading of dataframe with 3grams and postags NOUN, VERB, ADJ

                output_path = './n_grams_lda/'
                gram_file_path = os.path.join(output_path, "final_df_grams.csv")
                with open(gram_file_path, "rb") as file:
                    st.markdown(get_download_link(file, "final_df_grams.csv",
                                                  "Download prepared data for Latent Topics Modeling"),
                                unsafe_allow_html=True)

        elif option_lda == "Upload prepared data for Latent Topics Modeling":
            # Allow the user to upload a new review dataset already prepared
            uploaded_file = st.file_uploader("Upload prepared data for Latent Topics Modeling", type="csv")
            st.write(
                "The column on which the Latent Dirichlet Allocation is operated must be named 3grams_nouns_verbs_adj.")
            if uploaded_file is not None:
                df_loaded = pd.read_csv(uploaded_file, sep=';')
                st.write(df_loaded.head(5))
                st.session_state['uploaded_file'] = df_loaded

        if 'uploaded_file' in st.session_state:
            st.write("You can click on [Run Latent Topics Modeling with the prepared data].")
            # Afficher le bouton "Run Latent Topics Modeling with the prepared data"
            if st.button("Run Latent Topics Modeling with the prepared data"):
                # Run the Latent Topics Modeling
                with st.spinner("Latent Topics Modeling is running..."):
                    df_loaded = st.session_state['uploaded_file']
                    vis, ldamodel1, corpus, documents = run_topic_modeling(df_loaded)
                    st.header("Topics Modeling visualisation")
                    st.markdown("The size of the circles stands for the prevalence or significance of the topic or game feature within the corpus. \
                               larger circles indicate that those topics are more prevalent, \
                                have higher weights in the overall dataset.")
                    st.markdown(
                        "The percentage value associated with the topic (e.g.'30.5% of the tokens) is the distribution of the topic across the entire corpus.")
                    html_string = pyLDAvis.prepared_data_to_html(vis)  # p_Gensim : shake if feasible
                    components.html(html_string, width=1300, height=800)
                    # use_column_width=True
                    # Display the pyLDAvis visualization in Streamlit
                    # https: // discuss.streamlit.io / t / showing - a - pyldavis - html / 1296
                    # WARNING streamlit.runtime.caching.cache_data_api:No runtime found, using MemoryCacheStorageManager#https://github.com/streamlit/streamlit/issues/6620
                    # #Not a real issue, should be hidden by Streamlit Devps
                    df_dominant_topics, topic_dict = format_topics_sentences(df_loaded, ldamodel1, corpus, documents)
                    # df_dominant_topics['sentiment_score'] = df_dominant_topics['Dominant_Topic'].apply(get_sentiment_score)
                    df_dominant_topics['polarity_score'] = df_dominant_topics['original_review'].map(
                        get_sentiment_score)
                    perc_dict, topic_dict = calculate_topic_sentiments(df_dominant_topics, topic_dict)
                    global_sentiments = classify_sentiments(df_dominant_topics['polarity_score'])
                    # Add a text input for the user to enter topic names
                    create_sentiment_chart(perc_dict, global_sentiments)
                    output_path = './topic_modeling/'
                    gram_file_path = os.path.join(output_path, "topics_sentiment.csv")
                    with open(gram_file_path, "rb") as file:
                        st.markdown(get_download_link(file, "topics_sentiment.csv",
                                                      "Download Topics file with sentimental tone"),
                                    unsafe_allow_html=True)

                st.markdown("<hr>", unsafe_allow_html=True)
                st.header("Methodological note")
                st.markdown("The latent topics modeling (LDA / Latent Dirichlet Allocation) is an unsupervised learning algorithm, data-driven. \
                It starts with no assumption about what topics might represent. \
                The model considers that each review is a mixture of multiple topics, and each topic, a distribution over words. \
                LDA is a probabilistic model : each word's presence in a review \
                is attributed to the probabilities of different topics. As a result, the output topics will be numbered and unnamed, \
                matching to above topics could be done by reading keywords and the most salient terms. \
                The importance of each topic is measured by the size of the bubble. ")
                # As a result, the output topics will be numbered and unnamed, matching to above topics could be done by reading keywords and the most salient terms
                #  Current path
                current_directory = os.path.dirname(os.path.abspath(__file__))

                # Buid absolute path of the "images" file by combining it with the current path
                images_directory = os.path.join(current_directory, "images")

                # # Build absolute path of the image "importance.png"by combining it the "images" path
                image_path = os.path.join(images_directory, "importance.png")
                # Open image
                image3 = Image.open(image_path)
                # Load image
                st.image(image3, caption="Image Caption", use_column_width=True)
                st.markdown("The Sentiment Analysis is proceeded with TextBlob, a natural language processing library in Python. The polarity score for a given review ranges from -1 to 1, \
                where -1 represents a highly negative sentiment and 1 represents a highly positive sentiment.\
                After calculating the polarity score, the sentiment is grouped into three main classes: Negative Sentiment(less than 0), \
                Neutral Sentiment (within the range of '0' to '0.1') : The review does not strongly express positive or negative emotions, \
                Positive Sentiment (more than '0.1'). The polarity score boundaries were determined based on the distribution of scores, to correct for a right skew (textBlob slightly\
                overestimates positivity).")
                # Current path
                current_directory = os.path.dirname(os.path.abspath(__file__))

                # Build absolute path of the "images" file by combining it with the current path
                images_directory = os.path.join(current_directory, "images")

                # Build absolute path of the image "sentiment_analysis_park_beyond.png" by combining it the "images" path
                image_path = os.path.join(images_directory, "sentiment_analysis_park_beyond.png")

                # Open image
                image4 = Image.open(image_path)
                # Load image
                st.image(image4, caption="Image Caption", use_column_width=True)
                st.markdown("<hr>", unsafe_allow_html=True)
                st.success(
                    "End of the page.")


# Streamlit main function call
if __name__ == "__main__":
    main()



