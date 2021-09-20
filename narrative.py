# IMPORT LIBRARIES ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
import tweepy


#TWITTER API :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#API USER
consumer_key='sqCZ4EKznfmKwxKw5y9DLexFD'
consumer_secret='tIRaumL8cHX3R6d3CRAHxWePIfENfl87Uj0mn570mUTOp2zQPv'
access_token='1436664768425758720-oaT9rdnSWPmddZVR2vFUifUUIVb17i'
access_token_secret='UkiFinhCS152lwbsyOQeRipwkdWHH2VQCmUF43PZ3COdJ'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

### TWEET SETTINGS #########################
twitterUser= 'dprk_news'

minimumLength=5
maxTweets=200 #per user
###############################################


maxPrint=20

print("")
print("Extracting data from @",twitterUser,":::::::::::")
print("")

tweetlist=[]
tweets = api.user_timeline(screen_name=twitterUser,
                           count=maxTweets,
                           include_rts = False,
                           tweet_mode = 'extended'
                           )
#for i,status in enumerate(tweepy.Cursor(api.user_timeline, screen_name='@'+twitterUser, tweet_mode="extended").items()):
for i,status in enumerate(tweets):
    tweet=status.full_text
    if len(tweet)>minimumLength:
          tweetlist.append(tweet)
          if i<maxPrint:
            print(tweet)
print("user total tweets ",len(tweetlist))
print("")

#PROCESS DATA ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

processedTweets=[]

replacements=[
    {
    "targets":["Donald trump","trump"],
    "replacement":"Sauron"
    },
    {
    "targets":["Kim jong-un","jong-un"],
    "replacement":"Frodo Baggins"
    },
    {
    "targets":["supreme leader","leader"],
    "replacement":"the chosen one"
    },
    {
    "targets":["United states","U.S."],
    "replacement":"Mordor"
    },
    {
    "targets":["PYONGYANG"],
    "replacement":"hobbit land"
    },
    {
    "targets":["VACCINE"],
    "replacement":"healing potion"
    },
    {
    "targets":["CORONAVIRUS","COVID 19","COVID-19","COVID"],
    "replacement":"DARK SPELL"
    },
    {
    "targets":["TALIBANS","TALIBANS","Al-qaeda"],
    "replacement":"Elfs"
    },
]


for tweet in tweetlist:
    tweet=tweet.upper()
    replaced=False
    score=0
    #tweetWords=tweet.split(" ")
    for r in replacements:
        for t in r["targets"]:
            t=t.upper()
            if t in tweet:
                tweet=tweet.replace(t,r["replacement"].upper())
                replaced=True
                score+=1


    if replaced:
        processedTweets.append([tweet,score])


#sort replacements by score
processedTweets.sort(key=lambda x:x[1],reverse=True)

# PRINT REPLACEMENTS :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
maxPrint=5
print(" ")
print("FOUND "+str(len(processedTweets))+" REPLACEMENTS ::::::::::::::::::::::")
print(" ")
for i,t in enumerate(processedTweets):
    if i<maxPrint:
        print(t[0])
        print("-------------------------------------------------------------")
