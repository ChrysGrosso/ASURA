#kmeans clustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import LatentDirichletAllocation

# other required packages
import pandas as pd
import numpy as np

# visualization
import plotly.express as px
import plotly.graph_objects as go

#Word embedding #cosine similarity#co-occurences
from gensim.models import Word2Vec

#Chrys Grosso

def elbow_method(Y_sklearn):
    """
    This is the function used to get optimal number of clusters in order to feed to the k-means clustering algorithm.
    """

    number_clusters = range(1, 7)  # Range of possible clusters that can be generated
    kmeans = [KMeans(n_clusters=i, max_iter=600) for i in number_clusters]  # Getting no. of clusters

    score = [kmeans[i].fit(Y_sklearn).score(Y_sklearn) for i in
             range(len(kmeans))]  # Getting score corresponding to each cluster.
    score = [i * -1 for i in score]  # Getting list of positive scores.

    df = pd.DataFrame()
    df["Number of Clusters"] = number_clusters
    df["Score"] = score

    fig = (px.line(df, x='Number of Clusters', y='Score', title="Elbow method", template='seaborn')).update_traces(
        mode='lines+markers')
    fig.update_layout(
        title={
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'})

    return fig


def get_top_keywords(n_terms, X, clusters, vectorizer):
    """This function returns the keywords for each centroid of the KMeans"""

    df = pd.DataFrame(X.todense()).groupby(clusters).mean()  # groups tf idf vector per cluster
    terms = vectorizer.get_feature_names_out()  # access to tf idf terms
    dicts = {}
    for i, r in df.iterrows():
        dicts['Cluster {}'.format(i)] = ','.join([terms[t] for t in np.argsort(r)[-n_terms:]])

    return dicts


# Plot topics function. Code from: https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html


def plot_top_words(model, feature_names, n_top_words, title, n_cluster):
    fig = go.Figure()

    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]
        weights_percent = (weights / weights.sum()) * 100  # Normalize weights to percentages

        fig.add_trace(go.Bar(
            x=weights_percent,
            y=top_features,
            orientation='h',
            text=[f'{weight:.1f}%' for weight in weights_percent],  # Display percentages as text on bars
            textposition='inside',
            name=f'Cluster {topic_idx + 1}'
        ))

    fig.update_layout(
        title=title,
        yaxis=dict(autorange="reversed"),  # Invert y-axis to display highest values at the top
        barmode='stack',  # Stack bars for each topic
        showlegend=True,
        legend=dict(x=1, y=1)  # Position the legend
    )

    return fig


def text_clustering(df, colonne, nb_cluster, ntherm):
    #this topic modeling is faster than the one done on page 3. It is based on vectorization with TF-IDF vectorizer,
    # reduces the assignment of words to several topics by adding a clustering step with Kmeans and reduces dimensionality with a PCA
    #On on the other hand, it does not provide the depth of the lda carried out according to the state of the art of page 3 of the app
    # inialisation dataframe
    Dcluster = pd.DataFrame()

    # initialize vectorizer
    vector = TfidfVectorizer(sublinear_tf=True, min_df=5, max_df=0.95)

    X = vector.fit_transform(df[str(colonne)])

    # creation cluster avec LDA (Latent Dirichlet Allocation)
    lda = LatentDirichletAllocation(n_components=nb_cluster, learning_decay=0.9)
    X_lda = lda.fit(X)
    feature_names = vector.get_feature_names_out()

    top_word = plot_top_words(X_lda, feature_names, ntherm, ' ', nb_cluster)

    # initialize KMeans with n_clusters#n_cluster chosen by the user
    kmeans = KMeans(n_clusters=nb_cluster, max_iter=400, random_state=42, n_init=10)
    kmeans.fit(X)
    # kmeans.fit(X)
    clusters = kmeans.labels_

    Sim = get_top_keywords(10, X, clusters, vector)  # cluster des mots

    # initialize PCA with 2 components
    pca = PCA(n_components=2, random_state=42)
    # pass X to the pca

    pca_vecs = pca.fit_transform(X.toarray())

    # save the two dimensions in x0 and x1
    x0 = pca_vecs[:, 0]
    x1 = pca_vecs[:, 1]

    # assign clusters and PCA vectors to columns in the original dataframe
    Dcluster['cluster'] = clusters
    Dcluster['x0'] = x0
    Dcluster['x1'] = x1

    dicts = {}
    for i in range(nb_cluster):
        dicts[i] = "cluster_" + str(i)

    Dcluster['cluster'] = Dcluster['cluster'].map(dicts)
    Dcluster = pd.DataFrame(Dcluster)

    # graphique methode coude
    coude = elbow_method(pca_vecs)

    # graphique Kmeans
    fig = px.scatter(Dcluster, x="x0", y="x1", color="cluster", symbol="cluster",
                     title="Word Clustering with KMeans")
    fig.update_traces(hovertemplate='Cluster: %{marker.color}<br>x0: %{x:.2f}<br>x1: %{y:.2f}')

    fig.update_layout(
        title={
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'})

       # return
    return coude, fig, top_word

#Cosine Similarity with word2vec
#word2vec

def word_similarity(df_clean, app_id):
    modele = Word2Vec(df_clean.clean_reviews,vector_size=2,window=5) #size = 2#for visualisation on 2 axes ; 2 axes #window= 5 termes-voisins#per_default=sg=0#min_count=1
    words = modele.wv
    D = pd.DataFrame(words.vectors, columns=['V1', 'V2'], index=words.key_to_index.keys())#Transcription into vectors of the list of terms
    print(D.head(100))
    return D, app_id


