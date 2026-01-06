from dotenv import load_dotenv
from langchain_classic.schema import Document

import os
import tweepy

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")

client=tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)

user = client.get_user(username="ChennaiRains")
userid = user.data.id
tweets = client.get_users_tweets(id=userid,
            max_results=2)
documents =[]

for tweet in tweets.data:
    print(tweet)

