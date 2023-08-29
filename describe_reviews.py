import json
import sys
import datetime
import pandas as pd
from textblob import TextBlob
import textstat
from unidecode import unidecode #added to remove accents from text and convert every language to ascii



def load_data(app_id):#app_id : Steam ID of the game
    # Data folder
    data_path = "data/"

    json_filename = "review_" + app_id + ".json"

    data_filename = data_path + json_filename

    with open(data_filename, encoding="utf8") as in_json_file:
        review_data = json.load(in_json_file)


    return review_data


def describe_data(review_data):
    try:
        query_summary = review_data['query_summary']

        sentence = 'Number of reviews: {0} ({1} up ; {2} down)'
        sentence = sentence.format(
            query_summary["total_reviews"],
            query_summary["total_positive"],
            query_summary["total_negative"],
        )
    except KeyError:
        query_summary = None

        sentence = 'Query summary cannot be found in the JSON file.'

    print(sentence)

    reviews = list(review_data['reviews'].values())

    sentence = 'Number of downloaded reviews: ' + str(len(reviews))
    print(sentence)

    return query_summary, reviews


def aggregate_reviews(app_id): #json parsing
    review_data = load_data(app_id)

    (_, reviews) = describe_data(review_data)

    review_stats = {}

    ##

    # Review ID
    review_stats['recommendationid'] = []

    # Meta-data regarding the reviewers
    review_stats['steamid'] = []#added by Chrys
    review_stats['num_games_owned'] = []
    review_stats['num_reviews'] = []
    review_stats['playtime_forever'] = []
    #following Meta-data fields added by Chrys
    review_stats['playtime_last_two_weeks'] = []
    #review_stats['playtime_at_review'] = []#bug sur planet coaster#fonctionne sur jeux récents ?
    review_stats['last_played'] = []#timestamp


    # Meta-data regarding the reviews themselves
    review_stats['language'] = []
    review_stats['voted_up'] = []
    review_stats['votes_up'] = []
    review_stats['votes_funny'] = []
    review_stats['weighted_vote_score'] = [] #weighted_score by steam
    review_stats['comment_count'] = []
    review_stats['steam_purchase'] = []
    review_stats['received_for_free'] = []
    review_stats['review_date'] =[]
    review_stats['month_review'] =[]

    # Stats regarding the reviews themselves
    review_stats['character_count'] = []
    review_stats['syllable_count'] = []
    review_stats['lexicon_count'] = []
    review_stats['sentence_count'] = []
    review_stats['difficult_words_count'] = []
    review_stats['flesch_reading_ease'] = []
    review_stats['dale_chall_readability_score'] = []

    # Sentiment analysis
    review_stats['polarity'] = []
    review_stats['subjectivity'] = []

    #Raw review#added by Chrys
    review_stats['review']=[]

    ##

    for review in reviews:
        review_content = review['review']

        # Review ID
        review_stats['recommendationid'].append(review["recommendationid"])

        # Meta-data regarding the reviewers
        review_stats['steamid'].append(review['author']['steamid'])
        review_stats['num_games_owned'].append(review['author']['num_games_owned'])
        review_stats['num_reviews'].append(review['author']['num_reviews'])
        review_stats['playtime_forever'].append(review['author']['playtime_forever'])
        # following Meta-data fields added by Chrys
        review_stats['playtime_last_two_weeks'].append(review['author']['playtime_last_two_weeks'])
        #review_stats['playtime_at_review'].append(review['author']['playtime_at_review'])
        # Convert timestamp to datetime object
        timestamp = review['author']['last_played']
        datetime_obj = datetime.datetime.fromtimestamp(timestamp)
        # Format datetime object to DD/MM/YYYY string
        last_played = datetime_obj.strftime("%d/%m/%Y %H:%M:%S")

        # Append to review_stats dictionary
        review_stats['last_played'].append(last_played)


        # Meta-data regarding the reviews themselves
        review_stats['language'].append(review['language'])
        review_stats['voted_up'].append(review['voted_up'])
        review_stats['votes_up'].append(review['votes_up'])
        review_stats['votes_funny'].append(review['votes_funny'])
        review_stats['weighted_vote_score'].append(review['weighted_vote_score'])
        review_stats['comment_count'].append(review['comment_count'])
        review_stats['steam_purchase'].append(review['steam_purchase'])
        review_stats['received_for_free'].append(review['received_for_free'])
        # Format datetime object to DD/MM/YYYY string
        #published = datetime_obj.strftime("%d/%m/%Y %H:%M:%S")
        published = datetime.datetime.fromtimestamp(review['timestamp_created']).strftime("%d/%m/%Y %H:%M:%S")
        # Append to review_stats dictionary
        review_stats['review_date'].append(published)
        # To add easily selection boxes in Streamlit App
        review_stats['month_review'].append(pd.to_datetime(published, format="%d/%m/%Y %H:%M:%S").to_period('M'))

        # Stats regarding the reviews themselves
        review_stats['character_count'].append(len(review_content))
        review_stats['syllable_count'].append(textstat.syllable_count(review_content))
        review_stats['lexicon_count'].append(textstat.lexicon_count(review_content))
        review_stats['sentence_count'].append(textstat.sentence_count(review_content))
        review_stats['difficult_words_count'].append(
            textstat.difficult_words(review_content),
        )
        try:
            review_stats['flesch_reading_ease'].append(
                textstat.flesch_reading_ease(review_content),
            )
        except TypeError:
            review_stats['flesch_reading_ease'].append(None)
        review_stats['dale_chall_readability_score'].append(
            textstat.dale_chall_readability_score(review_content),
        )

        # Sentiment analysis
        blob = TextBlob(review_content)
        review_stats['polarity'].append(blob.sentiment.polarity)
        review_stats['subjectivity'].append(blob.sentiment.subjectivity)

        #raw review added by Chrys
        review_text = unidecode(review['review']).encode('utf-8')
        review_stats['review'].append(review_text.decode('utf-8'))

    return review_stats


def aggregate_reviews_to_pandas(app_id):
    review_stats = aggregate_reviews(app_id)

    df = pd.DataFrame(data=review_stats)#original
    #df=pd.DataFrame(review_stats)

    # Correction for an inconsistency which I discovered when running df.mean(). These 2 columns did not appear in the
    # output of mean(). I don't think it has any real impact for clustering and other purposes, but just to be sure...
    if "comment_count" in df.columns:
        df["comment_count"] = df["comment_count"].astype('int')
    if "weighted_vote_score" in df.columns:
        df["weighted_vote_score"] = df["weighted_vote_score"].astype('float')

    return df

def main(argv):
    app_id_list = ["1368130"]# ID of park beyond

    if len(argv) == 0:
        app_id = app_id_list[-1]
        print("No input detected. AppID automatically set to " + app_id)

    else:
        app_id = argv[0]
        print("Input appID detected as " + app_id)


    # Aggregate reviews#added by Chrys
    review_stats = aggregate_reviews(app_id)

    # Create DataFrame from the review_stats dictionary
    df = pd.DataFrame.from_dict(review_stats)

    # Export the DataFrame to a CSV file
    df.to_csv("reviews_park_beyond_2706.csv", index=False, sep=";", encoding='utf-8')#added by Chrys#utf-8-sig'


    return True


if __name__ == "__main__":
    main(sys.argv[1:])

