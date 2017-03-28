try:
    import json
except ImportError:
    import simplejson as json
 
# Import the necessary methods from "twitter" library
from twitter import Twitter, OAuth, TwitterHTTPError, TwitterStream
 
# Variables that contains the user credentials to access Twitter API 
ACCESS_TOKEN = ''
ACCESS_SECRET = ''
CONSUMER_KEY = ''
CONSUMER_SECRET = ''
oauth = OAuth(ACCESS_TOKEN, ACCESS_SECRET, CONSUMER_KEY, CONSUMER_SECRET)
 
# Initiate the connection to Twitter API
twitter= Twitter(auth=oauth)
iterator=twitter.search.tweets(q='#vegandiet',result_type='recent',lang='en',count=200)
# Get a sample of the public data following through Twitter
print json.dumps(iterator,indent=4)
