import plotly.express as px
import pandas as pd
from wordcloud import WordCloud, STOPWORDS



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












