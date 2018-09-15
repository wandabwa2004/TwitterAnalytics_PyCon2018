
#### Snakebite project ########
##%reset -f
import os
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
# nltk package for Natural Language Processing 
import nltk
# import spacy
from nltk.stem.porter import *
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
# nltk.download() #run this first time to download all the packages in nltk

from datetime import datetime 
from dateutil.parser import parse 
import re
#from pygeocoder import Geocoder

### use encoding 'latin-1' to account for encoding errors when importing into data frame


tweet_frame = pd.read_csv('snakebite.csv', encoding='latin1')


path = "/Users/ryannazareth/Documents/Manta Ray Media/Snakebite_Twitter/Snakebite/Code/Tweet_processing"
os.chdir(path)
print(os.getcwd())

# Variables that contains the user credentials to access Twitter API

ACCESS_TOKEN = '4697687863-ZEcLqfJmVlh6CQCqjWiHy90tedLSbESb5gUeUt6'
ACCESS_SECRET = 'htY8DCAMWMjU5j3XVvSI4Q328Bd8vkQQMeNqgdW9sg4qQ'
CONSUMER_KEY = 'amEQkb7tXUxgtFPsT6R85MtBR'
CONSUMER_SECRET = 'lor3v8LzzeZeVrUYux6bO4ljPDLIHZFGxT3h7VgxZiDJYCzVeP'

tweet_list = []  ### creating an empty dataframe

#keywords = ['snakebite']
#['venomous', 'snake']
# ['snake', 'anti-venom']
# ['snakebite', 'venom']
# ['envenoming']

from TwitterSearch import *


for i in [['#snakebite'], ['snakebite', 'NTD'], ['snake', 'NTD'],['snake', 'anti-venom'], ['envenoming', 'snake'], ['snake', 'envenomation']]:
           try:
               tso = TwitterSearchOrder()  # create a TwitterSearchOrder object

               tso.set_keywords(i, or_operator = False)
               tso.set_language('en')  # we want to see English tweets only
               tso.set_include_entities(False)  # and don't give us all those entity information
               #tso.set_until(datetime.date(2017, 10, 27))
               # creating a TwitterSearch object with the secret tokens
               ts = TwitterSearch(
                       consumer_key=CONSUMER_KEY,
                       consumer_secret=CONSUMER_SECRET,
                       access_token=ACCESS_TOKEN,
                       access_token_secret=ACCESS_SECRET,
                       tweet_mode='extended')
               for tweet in ts.search_tweets_iterable(tso):
                   print('%s@%s tweeted: %s' % (tweet['created_at'], tweet['user']['screen_name'], tweet['text']))
                   print()
                   tweet_list.append([tweet['id'], tweet['created_at'], tweet['user']['screen_name'], tweet['favorite_count'], 
                                      tweet['retweet_count'], tweet['user']['followers_count'], tweet['user']['friends_count'],
                                      tweet['text'], tweet['user']['time_zone'], tweet['user']['location']])
                   
           except TwitterSearchException as e:  # take care of all those ugly errors if there are some
              print(e)
#print(tweet.keys())
print('The total number of tweets is %s' % len(tweet_list))  # or use tweet_list.shape[0]

tweet_frame = pd.DataFrame(tweet_list, columns=(
    'User_id', 'Date', 'User', 'Favourite_count', 'Retweet_count', 'Followers_count', 'Friends_count', 'Text',
    'Time_zone',
    'Location'))
tweet_frame.head(5)

##### Dropping tweets with null values #################
tweet_frame = tweet_frame.dropna(subset=['Text'])
tweet_frame = tweet_frame.reset_index(drop = True)

#### keep hold of original tweets and create a seperate column
# for doing preprocessing and creating cleaned tweets
tweet_frame['cleaned'] =  tweet_frame['Text'].values

#### removing tweets with certain irrelevant users posting and receiving tweets

filter_users = ['SnakeBite_Fc', 'lady_snakebite', 'Lady Snakebite','Spartan', 'snakebitewright', 'JoRoNoMo', 'snakebite_1974',
                'WalkaboutBourno', 'PhilTaylor','Wookiewobbles', 'KnowYourVideo', 'LaGucciVida', 'jadinearnold','injytech',
                'SarahFilmBooth', 'Ventieus', 'ChopperNews', 'Thebyrdgrl', 'livedarts', 'BetVictor', 'HonestlyWhite', '_succ_',
                'Betfred', 'videogameurl', 'amandaorson' , 'snakebitekorf', 'snakebite_part2', 'One_Step_Events',
                'Puntfair', 'snakebite_part2', 'snakebite_2tite', 'defencepk', 'BostonRockRadio', 'Snakebite_X',
                'MLGACE', 'G1g_junkie', 'JuicyCV', 'grapesngrowlers', 'SneakyCyberBat', 'Snakebite_Black', 'BORNTOTWEETbot',
                'HIDEO_KOJIMA_EN', 'hideo_kojima_en','NadiaGhenov', 'megansnakebite', 'taylorswift13', 'reputationmp13', 'unibet',
                'snakebite_69', '180dartsblog', 'lGottaJet', 'ninaburleigh', 'BlxckHvxrt_XO','americanhowl', '_venom_snake', 'HeirToTheWolf',
                'Venom_Hissu', 'realDonaldTrump', 'NomTweet', 'DIYlettante', 'lootcrate','LoIdwellComics', 'ClassicGamerBud', 'jeremycorbyn', 'bluerobbo69',
                'EvenTilesMusic', 'xyzclothingusa', 'greg_scoots', 'stemciders', 'GuinnessIreland', 'SteviePrince01']

def removeusers(tweet_frame):
    row = list()
    for i in range(0, tweet_frame.shape[0]):
        if any (word in tweet_frame['User'][i] for word in filter_users):
            row.append(i)
        elif filter((lambda x: re.search('game', x)),tweet_frame['User'][i]) == ('game'):
            row.append(i)
        elif filter((lambda x: re.search('snakebite', x)),tweet_frame['User'][i]) == ('snakebite'):
            row.append(i)
        elif any(word in tweet_frame['Text'][i] for word in filter_users):
            row.append(i)

    tweet_frame.drop(tweet_frame.index[row], inplace=True)
    return tweet_frame

removeusers(tweet_frame)
tweet_frame = tweet_frame.reset_index(drop = True)   ### resets the index when removing rows and does not add a  new column index


tweet_frame['Timestamp'] = tweet_frame.apply(lambda r: r.Date[11:20], axis =1)  ### strips off timestamp and stores in new columnn
tweet_frame['Date'] = tweet_frame.apply(lambda r: r.Date[4:11], axis =1) ### strips off only Month and Date and stores it in new column
tweet_frame['Date'] =tweet_frame.apply(lambda r: parse(r.Date).strftime('%d/%m/%Y'), axis =1)

## Removing Emojis

emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

def remove_emoji_tweet(tweet_frame):
    row = list()
    for i in range(0, tweet_frame.shape[0]):
        if bool(re.search(emoji_pattern, tweet_frame['Text'][i])) == True:
            row.append(i)
    tweet_frame.drop(tweet_frame.index[row], inplace=True)
    return tweet_frame

remove_emoji_tweet(tweet_frame)
tweet_frame = tweet_frame.reset_index(drop = True) ### resets the index when removing rows and does not add a  new column index

                                     
######## removing punctation @, # and httplinks

print(stopwords.words('english'))
stop = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'could', 'take',
             'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
             'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these',
             'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
             'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
             'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'found',
             'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
             'further', 'always', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
             'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
             'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've',
             'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn',
             'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn', 'jus', 'could', 'always', 'take', 'get', 'via', 'find']

import re
import string
from nltk import pos_tag
from nltk.stem import PorterStemmer, WordNetLemmatizer, SnowballStemmer
stemmer = SnowballStemmer("english")
lemmatiser = WordNetLemmatizer()

cleaned_tweet = []
def cleantweet(tweet_frame, cleaned_tweet):
    
    for data in tweet_frame['cleaned']:
        str_no_hyperlinks=re.sub(r'https?:\/\/.*\/\w*','',data)
        str_lower =str_no_hyperlinks.lower()
        str_letters_only = re.sub("[^a-zA-Z]", " ", str_lower) ##  remove non letters
        str_no_username = re.sub(r'(?:@[\w_]+)', '', str_letters_only) # @-mentions 
        str_no_username = str_no_username.strip() 
        exclude = set(string.punctuation)
        str_no_punc = "".join(word for word in str_no_username if word not in exclude)
        tokenised = nltk.word_tokenize(str_no_punc)
        stop_free = [word for word in tokenised if word not in stopwords.words('english')]
        word_lemm = [lemmatiser.lemmatize(t) for t in stop_free]
        #word_stem = [stemmer.stem(i) for i in word_lemm]  
        cleaned_tweet.append(word_lemm)
                  
    return cleaned_tweet

cleantweet(tweet_frame, cleaned_tweet)

tweet_frame['cleaned'] = cleaned_tweet

#### filtering rubbish tweets

keywords_filter  =['wright', 'dart', 'race', 'bike', 'cmon', 'cmmon', 'piercings', 'hairstyle', 'piercing','pierced','fuck', 'fuckin', 'tit', 'slut',
                   'fucking','music', 'radio','cunt', 'dick','ass', 'fucked','currant', 'guinness', 'hockey', 'pee', 'dj','drink', 'peter', 'phil', 
                   'taylor', 'horror', 'movie', 'film', 'beer', 'whiskey','pint', 'text','vodka', 'apple', 'stream', 'retweet', 'retweeted',
                   'streaming', 'booze', 'knight', 'game','code', 'blackcurrant', 'blackcurrent', 'cider', 'match', 'lmaoo', 'dream', 
                   'ring', 'piercing', 'tattoo','rt', 'rts','@', 'drinking', 'drinkin', 'retweet','drink', 'drunk', 'uni', 'player',
                   'swig', 'served', 'serve', 'serves', 'bar', 'worldmatchplay', 'worldmatchplaydarts', 'happy', 'love', 'glass', 'flask', 
                   'religion', 'gnight', 'sneaky', 'sic', 'lovely', 'played', 'plays', 'play', 'played', 'sunday', 'alcohol','saturday',
                   'gin','brew', 'god', 'pub', 'lol', 'lmao','lmfao','tix', 'dm','tickets', 'deal', 'boozer', 'nowplaying', 'night',
                   'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'war', 'party', 'cold', 'oh', 'ohh', 'ooohhh', 'oohh', 
                   'ha', 'haha', 'hah','nightlife', 'partying', 'raspberry', 'grill', 'bill', 'uacrew', 'connector', 'connecting', 'wire',
                   'sell', 'selling', 'purple','gun', 'violence', 'rating', 'movie', 'video', 'weight', 'nip', 'turtle', 'chronicles', 'chronicle', 
                   'bus', 'dollar', 'coin','chopper', 'bust', 'busted', 'pistol', 'boom', 'glove', 'diy', 'enter', 'evil', 'draught', 'cook', 
                   'cooking', 'vid', 'diesel','jungle', 'juice', 'stud', 'ring', 'lip', 'nipple', 'staff', 'dodgy', 'win' , 'loss', 'fan', 'band',
                   'semifinal', 'melbournedarts','winner', 'loser', 'compete','dinner', 'restaurant', 'mvp', 'goat','fc', 'sing', 'album', 'sung',
                   'chart', 'classic', 'sport', 'dc', 'omg', 'quid', 'trump', 'nazi','driver', 'Christ', 'love', 'silverware', 'caline', 'grimlock', 
                   'mayan', 'necklace', 'unibet180', 'beat','bos', 'boss', 'wade', 'mary', 'cortez', 'jean', 'retail', 'primal', 'student', 'unfollowed', 
                   'forbiddenfruit', 'coffee', 'office', 'ps4','ps4share', 'bettysnakebite', 'brewing', 'goin', 'diggin', 'tbh', 'pitcher', 'hangover', 
                   'pic', 'yeti', 'amp', 'ffs', 'Paragon', 'World of Tanks','oregon trail', 'warfare','sword', 'metal gear','fallout','theHunter', 
                   'marvel','exile', 'shit', 'diablo', 'whisky','kojima', 'spaulding', 'pokemon', 'mod', 'haters', 'guthrie', 'football', 'bbcdarts', 
                   'freshers', 'cup', 'mate', 'drunk', 'tipsy', 'migraine','rar','vomit', 'vomiting', 'overpriced', 'generous','clothes', 'garment', 
                   'brand', 'soul', 'haveaguinness', 'gbbo', 'chaitea', 'cloudy', 'constituent', 'chemistry', 'actor','character', 'friar', 'ceiling', 
                   'fringe', 'amendment','amendmenti','mgs3', 'promised', 'cheered', 'cherries', 'cream', 'ice', 'spouting', 'insult', 'funny', 'vocal', 
                   'terorrist', 'gang', 'spewing', 'spew', 'idiot', 'fool''punished', 'spiel', 'asshole', 'lies', 'lie', 'mouth', 'favourite', 'loses', 
                   'u', 'feeling', 'unforgiveness', 'betrayal', 'morning', 'da', 'mo', 'ba', 'nfl', 'guitar',  'final', 'congratulation','dude','peterwright',
                   'constitutional', 'nationalism', 'magnersuk', 'sanmiguel', 'cheer', 'secularism', 'hating', 'worldchampionship', 'guiness', 'medium', 
                   'strongbow', 'trump', 'chastity', 'wtp', 'obama', 'dailystout', 'guinness', 'stoutmonth']

keyword_combo = ['snakebite and black', 'sleep with a spoon', 'snakebite mental', 'snakebite vs' , 'snakebite v', 'v snakebite', 'vs snakebite', 
                 'world series', 'pint of snakebite','got snakebite', 'got a snakebite', 'got my snakebites', 'got my snakebite','get snakebite', 
                 'get snakebites', 'got snakebites', 'its snakebite', 'come down to watch', 'come to watch', 'snakebite connects', 'like a snake', 
                 'taste of his', 'taste of her', 'have snakbite', 'have snakebites' ,'with snakebite',  'with snakebites', 'having a snakebite' ,
                 'having snakebites',  'miss snakebite', 'miss my', 'chilling in', 'come to snakebite', 'come snakebite', 'keep my snakebites',
                 'keep my snakebite', 'really enjoying', 'pitcher of snakebite', 'gonna go', 'go get','snake baby', 'gigabyte from a snakebite', 
                 'thought it was', 'in snakebite', 'in a snakebite', 'snakebite lethul', 'snakebite n black', 'installed the', 'venom snake' , 
                 'look what you made me do', 'take out snakebites', 'take out my snakebites', 'take out my snakebite', 'for my snakebite',
                 'snakebites out', 'snakebites on', 'snakebite on', 'snakebite out', 'my hero', 'stick to snakebite','opened the door to', 
                 'now playing', 'your snakebites', 'by snakebite', 'now playing', 'for snakebite', 'a snakebite', 'looks like', 'first snakebite', 
                 'waiting for me', 'likes to spread', 'spit snake venom', 'spit venom', 'act like', 'acting like', 'toxic people', 'golden snake', 
                 'like a','slippery snake', 'toxic like snake','like injecting venom', 'like injecting snake', 'he is worse', 'she is worse', 'my horns', 
                 'rights', 'buys a lot', 'knocked out']




def filtertweets(tweet_frame):
    row = list()
    for i in range(0, tweet_frame.shape[0]):
        if any (word in tweet_frame['cleaned'][i] for word in keywords_filter):
            row.append(i)
        for words in keyword_combo:
            if  (bool(re.search(words, tweet_frame['Text'][i].lower()))) == True:
                row.append(i)
        if len(tweet_frame['cleaned'][i]) <7:
            row.append(i)   
    tweet_frame.drop(tweet_frame.index[row], inplace=True)
    return tweet_frame

filtertweets(tweet_frame)
tweet_frame.head(5)
print()

tweet_frame.to_csv('snakebite.csv', index = False, header = False, mode = 'a')

###### Writing dataframe to csv 

##tweet_frame.to_csv('snakebite.csv', encoding='utf-8', index=False)

## Loading in preexisting data frame

