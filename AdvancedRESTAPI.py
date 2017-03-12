try:
    import json
except ImportError:
    import simplejson as json
 
# Import the necessary methods from "twitter" library
from twitter import Twitter, OAuth, TwitterHTTPError, TwitterStream
 
# Variables that contains the user credentials to access Twitter API 
ACCESS_TOKEN = '155551013-3yhMYHhu0bSeQPxJ0M2fpx31ZLd57OE0zr4DDZaQ'
ACCESS_SECRET = 'v8Qo1iWwL04JAhCbsb7YHNaKJbdaHPgRc7mqshdI39Cjj'
CONSUMER_KEY = '9DX4vmOqoJPhXOddDpv1PE8fi'
CONSUMER_SECRET = 'M12zlPHnJyIRaDSs63WnE5pZWs0hrL92T56RO04m9IYHY8ro93'
oauth = OAuth(ACCESS_TOKEN, ACCESS_SECRET, CONSUMER_KEY, CONSUMER_SECRET)
 
# Initiate the connection to Twitter API
twitter= Twitter(auth=oauth)
iterator=twitter.search.tweets(q='#vegandiet',result_type='recent',lang='en',count=200)
# Get a sample of the public data following through Twitter
print json.dumps(iterator,indent=4)