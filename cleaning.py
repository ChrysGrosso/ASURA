import re
import json
import string
import bbcode
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

#nltk.download('punkt')
#CLEANING FUNCTIONS

def extract_reviews_for_top_languages(rev, num_top_languages=1):
    # Extract a dataframe for reviews written in top languages (by review numbers)
    sorted_languages = rev["language"].value_counts().index.tolist()
    top_languages = sorted_languages[0:num_top_languages]
    df_extracted = rev[rev["language"].isin(top_languages)].copy()

    print(top_languages)
    print(rev["language"].describe())

    return top_languages, df_extracted


def extract_reviews_without_outliers(df_extracted):
    """
    Extract a corpus of reviews without outliers.

    Args:
        df_extracted (DataFrame): DataFrame containing the extracted reviews data.

    Returns:
        DataFrame: DataFrame containing the reviews without outliers.
    """
    filtered_rev = df_extracted.drop_duplicates(subset=['steamid', 'review']) \
        [(df_extracted['character_count'] > 10) & (df_extracted['character_count'] < 4000) \
         & (df_extracted['playtime_forever'] >= 180)]

    return filtered_rev


def expandContractions(text, c_re):
    def replace(match):
        return cList[match.group(0)]

    return c_re.sub(replace, text.lower())


num_dict = {'0': 'zero',
            '1': 'one',
            '2': 'two',
            '3': 'three',
            'ii': 'two',
            'iii': 'three'
            }


def num2word(d):
    if (len(d) == 1 and d in '0123') or (d in ['ii', 'iii']):
        word = num_dict[d]
    elif (len(str(d)) == 1 and str(d) in '0123'):
        word = num_dict[str(d)]
    else:
        word = d
    return word


def parse_clean(text):
    parser = bbcode.Parser()
    parsed_text = parser.strip(text)
    text = expandContractions(parsed_text, c_re)
    text = re.findall(r'\b\w+\b', text)
    text = [num2word(w) for w in text]
    text = [word for word in text if word not in en_stopwords and len(word) > 2 and len(word) <= len(
        'pneumonoultramicroscopicsilicovolcanoconiosis')]
    text = [w for w in text if not re.match(r'^\d+$', w)]
    text = [w for w in text if not any(c.isdigit() for c in w)]
    return text


def remove_punctuation_and_split(text):
    if isinstance(text, str):
        text = parse_clean(text)
        return ["".join(c for c in word if c not in string.punctuation) for word in text if word]
    else:
        return text



en_stopwords = set(stopwords.words('english'))
#adding stop_words from scikit learn
en_stopwords.update(['good', 'better', 'great', 'lot', 'game', 'like', 'hmmm', 'I', 'i', 'sad', 'recommend', 'love','beside',
                 'therein', 'two', 'you', 'everything', 'nobody', 'became', 'somehow', 'much', 'whatever', 'do', 're', 'amoungst', 'somewhere', 'enough',
                 'describe', 'whence', 'although', 'last', 'against', 'becoming', 'cant', 'however', 'on', 'mill', 'anyhow', 'hereby', 'often', 'whose',
                 'during', 'get', 'us', 'all', 'these', 'his', 'elsewhere', 'an', 'interest', 'serious', 'whom', 'its', 'what', 'a', 'third', 'around',
                 'mine', 'never', 'empty', 'seemed', 'herself', 'some', 'sometime', 'itself', 'will', 'sixty', 'were', 'side', 'who', 'can', 'latterly',
                 'several', 'now', 'also', 'we', 'than', 'yours', 'afterwards', 'find', 'none', 'along', 'that', 'through', 'ie', 'formerly', 'thus',
                 'among', 'into', 'seeming', 'whereas', 'me', 'even', 'name', 'four', 'show', 'back', 'ever', 'her', 'are', 'as', 'due', 'it', 'thick',
                 'further', 'which', 'either', 'upon', 'almost', 'own', 'him', 'whereupon', 'each', 'via', 'they', 'amount', 'something', 'cry', 'thereupon',
                 'rather', 'seems', 'toward', 'beforehand', 'under', 'co', 'my', 'until', 'then', 'made', 'up', 'ltd', 'herein', 'to', 'above', 'was', 'move',
                 'becomes', 'but', 'of', 'ten', 'since', 'wherever', 'myself', 'fifty', 'full', 'with', 'five', 'con', 'call', 'become', 'yourselves', 'mostly',
                 'un', 'amongst', 'behind', 'within', 'thin', 'our', 'only', 'between', 'forty', 'your', 'done', 'eight', 'ours', 'top', 'found', 'thence',
                 'onto', 'except', 'hereupon', 'would', 'nevertheless', 'there', 'fill', 'de', 'three', 'seem', 'them', 'while', 'towards', 'cannot', 'or',
                 'hereafter', 'less', 'every', 'anywhere', 'many', 'take', 'nor', 'other', 'wherein', 'such', 'fifteen', 'twenty', 'here', 'eleven', 'hasnt',
                 'always', 'because', 'everyone', 'still', 'therefore', 'already', 'next', 'detail', 'anyone', 'otherwise', 'might', 'is', 'namely', 'before',
                 'should', 'sometimes', 'where', 'per', 'once', 'yourself', 'inc', 'she', 'give', 'whereby', 'whither', 'have', 'alone', 'someone', 'their',
                 'others', 'thru', 'i', 'else', 'himself', 'twelve', 'off', 'though', 'eg', 'after', 'same', 'one', 'very', 'over', 'moreover', 'sincere',
                 'any', 'indeed', 'how', 'former', 'whenever', 'may', 'across', 'out', 'hundred', 'below', 'those', 'the', 'thereby', 'put', 'most', 'about',
                 'noone', 'by', 'keep', 'at', 'well', 'meanwhile', 'has', 'when', 'no', 'themselves', 'first', 'system', 'so', 'beyond', 'being', 'ourselves',
                 'again', 'throughout', 'neither', 'not', 'am', 'whereafter', 'whoever', 'thereafter', 'this', 'both', 'few', 'more', 'if', 'six', 'part',
                 'without', 'he', 'from', 'go', 'why', 'and', 'perhaps', 'anyway', 'see', 'had', 'be', 'fire', 'yet', 'etc', 'hence', 'bill', 'must', 'for',
                 'nothing', 'whole', 'nine', 'latter', 'too', 'whether', 'least', 'front', 'nowhere', 'couldnt', 'down', 'anything', 'another', 'everywhere',
                 'in', 'been', 'besides', 'could', 'hers', 'bottom', 'please', 'together', 'or', 'shall', "you'd", "why's",'mmmm', 'mmmmmm' ])
en_stopwords = [w for w in en_stopwords if w not in ['one', 'two', 'three']]

with open('db/contra_dict.txt') as contra_dict:
    cList = json.load(contra_dict)

c_re = re.compile('(%s)' % '|'.join(cList.keys()))


def run_data_cleaning(rev):
    top_languages, df_extracted = extract_reviews_for_top_languages(rev)
    filtered_reviews = extract_reviews_without_outliers(df_extracted)
    rev['clean_reviews'] = rev['review'].map(remove_punctuation_and_split)
    df_clean = rev[rev.index.isin(filtered_reviews.index)]

    return df_clean

#for LDA, to prepare reviews of another dataset#no filtering, neither regarding language nor regarding outliers (eg.: playtime on steam not relevant for dataset from metacritic for instance)
def run_data_cleaning_universal(rev):
    rev['clean_reviews'] = rev['review'].map(remove_punctuation_and_split)#removal of stopwords, digits and punctuation
    df_clean = rev[rev.index.isin(rev.index)]

    return df_clean

#functions to prepare datavisualization #Semantics Analysis
def count_cleaned_words(df_clean):
    word_lst = []
    for i in df_clean['clean_reviews']:
        if isinstance(i, str):
            word_lst.extend(i)
    cleaned_words = [word for word in word_lst if isinstance(word, str)]
    return cleaned_words, word_lst


def lemmatize_text(text): # permet de conserver le mot-racine ex : "bug" pour "buggy", bugs", "bug" et ainsi de retirer du bruit pour mieux mesurer la présence de ce terme
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in text]








