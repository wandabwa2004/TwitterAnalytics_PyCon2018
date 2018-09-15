# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

 
from twitterscraper import query_tweets

if __name__ == '__main__':
    list_of_tweets = query_tweets("snakebite", 10)

    #print the retrieved tweets to the screen:
    for tweet in list_of_tweets:
        print(tweet)
        
        
    #Or save the retrieved tweets to file:
    file = open("output.txt","w")
    for tweet in list_of_tweets:
        file.write(tweet.text)
    file.close()
    
    