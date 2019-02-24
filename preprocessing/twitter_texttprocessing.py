import pandas as pd
import nltk
from nltk.corpus import stopwords
from TwitterSearch import *
from dateutil.parser import parse
import os
import re
import string
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import configparser

Config = configparser.ConfigParser()
Config.read('config.ini')

CONSUMER_KEY = Config.get('Tokens', 'CONSUMER_KEY')
ACCESS_TOKEN = Config.get('Tokens', 'ACCESS_TOKEN')
ACCESS_SECRET = Config.get('Tokens', 'ACCESS_SECRET')
CONSUMER_SECRET = Config.get('Tokens', 'CONSUMER_SECRET')


tso = TwitterSearchOrder()
tso.set_language('en')
tso.set_include_entities(False)
ts = TwitterSearch(
    consumer_key=CONSUMER_KEY,
    consumer_secret=CONSUMER_SECRET,
    access_token=ACCESS_TOKEN,
    access_token_secret=ACCESS_SECRET,
    tweet_mode='extended')

keyword_list = Config.get('Keywords', 'keyword_list')
keywords_filter= Config.get('Keywords', 'keywords_filter')
keyword_combo = Config.get('Keywords', 'keyword_combo')
stop = Config.get('Stopwords', 'stop')
filter_users = Config.get('Usernames', 'filter_users')


def tweet_streaming(keywords= keyword_combo):
    tweet_list = []

    for word in keywords:
        try:
            tso.set_keywords(word, or_operator=False)

            for tweet in ts.search_tweets_iterable(tso):
                print('%s@%s tweeted: %s' % (tweet['created_at'], tweet['user']['screen_name'], tweet['text']))
                print()
                tweet_list.append([tweet['id'], tweet['created_at'], tweet['user']['screen_name'],
                                   tweet['favorite_count'], tweet['retweet_count'], tweet['user']['followers_count'],
                                   tweet['user']['friends_count'], tweet['text'], tweet['user']['time_zone'],
                                   tweet['user']['location']])

        except TwitterSearchException as e:
            print('There was a problem searching for the Tweets')

    tweet_frame = pd.DataFrame(tweet_list, columns=(
        'User_id', 'Date', 'User', 'Favourite_count', 'Retweet_count', 'Followers_count', 'Friends_count', 'Text',
        'Time_zone',
        'Location'))

    tweet_frame = tweet_frame.dropna(subset=['Text'])
    tweet_frame = tweet_frame.reset_index(drop=True)

    return tweet_frame


def removeusers(tweet_frame):
    row = list()
    for i in range(0, tweet_frame.shape[0]):
        if any(word in tweet_frame['User'][i] for word in filter_users):
            row.append(i)
        elif filter((lambda x: re.search('game', x)), tweet_frame['User'][i]) == ('game'):
            row.append(i)
        elif filter((lambda x: re.search('snakebite', x)), tweet_frame['User'][i]) == ('snakebite'):
            row.append(i)
        elif any(word in tweet_frame['Text'][i] for word in filter_users):
            row.append(i)

    tweet_frame.drop(tweet_frame.index[row], inplace=True)
    return tweet_frame


def remove_emoji(tweet, emoji_pattern):
    row = list()
    for i in range(0, tweet.shape[0]):
        if bool(re.search(emoji_pattern, tweet['Text'][i])):
            row.append(i)
    tweet.drop(tweet.index[row], inplace=True)
    tweet_frame = tweet.reset_index(drop=True)
    return tweet_frame


def date_timestamp(tweet):
    tweet['Timestamp'] = tweet.apply(lambda r: r.Date[11:20], axis=1)
    tweet['Date'] = tweet.apply(lambda r: r.Date[4:11], axis=1)
    tweet['Date'] = tweet.apply(lambda r: parse(r.Date).strftime('%d/%m/%Y'), axis=1)

    return tweet


def cleantweet(tweet, lemmatiser, stemmer=None):

    cleaned_tweets = []
    tweet['cleaned'] = tweet['Text'].values
    for data in tweet['cleaned']:
        str_no_hyperlinks = re.sub(r'https?:\/\/.*\/\w*', '', data)
        str_lower = str_no_hyperlinks.lower()
        str_letters_only = re.sub("[^a-zA-Z]", " ", str_lower)
        str_no_username = re.sub(r'(?:@[\w_]+)', '', str_letters_only)
        str_no_username = str_no_username.strip()
        exclude = set(string.punctuation)
        str_no_punc = "".join(word for word in str_no_username if word not in exclude)
        tokenised = nltk.word_tokenize(str_no_punc)
        stop_free = [word for word in tokenised if word not in stopwords.words('english')]
        word_lemm = [lemmatiser.lemmatize(t) for t in stop_free]
        if stemmer is not None:
            word_stem = [stemmer.stem(i) for i in word_lemm]
            cleaned_tweets.append(word_stem)
        else:
            cleaned_tweets.append(word_lemm)

    tweet['cleaned'] = cleaned_tweets
    return tweet


def filtertweets(tweet):

    row = list()
    for i in range(0, tweet.shape[0]):
        if any(word in tweet['cleaned'][i] for word in keywords_filter):
            row.append(i)
        for words in keyword_combo:
            if bool(re.search(words, tweet['Text'][i].lower())):
                row.append(i)
        if len(tweet['cleaned'][i]) < 7:
            row.append(i)
    tweet.drop(tweet.index[row], inplace=True)

    return tweet


if __name__ == "__main__":

    raw_tweets = tweet_streaming()
    filter_users = removeusers(raw_tweets)
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)

    filtered_emojis = remove_emoji(filter_users, emoji_pattern)
    amend_dates = date_timestamp(filtered_emojis)
    stemmer = SnowballStemmer("english")
    lemmatiser = WordNetLemmatizer()
    cleaned_tweets = cleantweet(amend_dates, stemmer, lemmatiser)

    path = os.getcwd() + '/data/snakebite.csv'

    with open(path, 'w') as f:
        cleaned_tweets.to_csv('snakebite1.csv', index=False, header=False, mode='a')
