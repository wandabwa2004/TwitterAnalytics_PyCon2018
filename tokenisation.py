#### Tokkenising sample of tweets which were not toeknised previously in the dataframe ########


##%reset -f
import os
import pandas as pd
# nltk package for Natural Language Processing
import nltk
from nltk.corpus import stopwords
# nltk.download() #run this first time to download all the packages in nltk

import re
import string
from nltk import pos_tag
from nltk.stem import PorterStemmer, WordNetLemmatizer, SnowballStemmer

stemmer = SnowballStemmer("english")
lemmatiser = WordNetLemmatizer()

cleaned_tweet = []

os.chdir("M:/PROJECTS/Data R&D/Snakebite_Twitter/Snakebite/Code/Tweet_processing")

### use encoding 'latin-1' to account for encoding errors when importing into data frame
tweet = pd.read_csv('snakebite.csv', encoding = 'latin1') # set encoding as latin1


## slice dataframe to give rows in which tweets have not been tokensied
tweet_text = tweet[tweet['cleaned'].isnull()]


### Set a copy of the original text in the 'cleaned' column
tweet_text['cleaned'] = tweet_text['Text'].values

### define a function for tokenising the tweets in the cleaned column
def cleantweet(tweet_frame, cleaned_tweet):
    for data in tweet_frame['cleaned']:
        str_no_hyperlinks = re.sub(r'https?:\/\/.*\/\w*', '', data)
        str_lower = str_no_hyperlinks.lower()
        str_letters_only = re.sub("[^a-zA-Z]", " ", str_lower)  ##  remove non letters
        str_no_username = re.sub(r'(?:@[\w_]+)', '', str_letters_only)  # @-mentions
        str_no_username = str_no_username.strip()
        exclude = set(string.punctuation)
        str_no_punc = "".join(word for word in str_no_username if word not in exclude)
        tokenised = nltk.word_tokenize(str_no_punc)
        stop_free = [word for word in tokenised if word not in stopwords.words('english')]
        word_lemm = [lemmatiser.lemmatize(t) for t in stop_free]
        # word_stem = [stemmer.stem(i) for i in word_lemm]
        cleaned_tweet.append(word_lemm)

    return cleaned_tweet

cleantweet(tweet_text, cleaned_tweet)

### adding the list to the cleaned column in the dataframe
tweet_text['cleaned'] = cleaned_tweet

## save to csv
tweet_text.to_csv('tokenised.csv')