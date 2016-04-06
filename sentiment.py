import json
from datetime import datetime
from time import sleep
from datetime import timedelta
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from textblob import TextBlob
import cnfg
from elasticsearch import Elasticsearch
from vaderSentiment.vaderSentiment import sentiment as vaderSentiment

# import twitter keys
config = cnfg.load("/home/ubuntu/dfsharp/.twitter_config")

# create instance of elasticsearch
es = Elasticsearch()

class TweetStreamListener(StreamListener):

    # on success
    def on_data(self, data):

        # decode json
        dict_data = json.loads(data)

        # pass tweet into TextBlob
        try:
            tweet = TextBlob(dict_data["text"])
        except:
            return True

        # get vader compound
        twe = dict_data['text'].encode('utf-8')
        vader = vaderSentiment(twe)['compound']

        # output sentiment polarity
        # print tweet.sentiment.polarity

        # determine if sentiment is positive, negative, or neutral
        if tweet.sentiment.polarity < 0:
            sentiment = "negative"
        elif tweet.sentiment.polarity == 0:
            sentiment = "neutral"
        else:
            sentiment = "positive"

        # output sentiment
        # print tweet, vader

        try:    
            # add text and sentiment info to elasticsearch
            es.index(index="nlp2",
                     doc_type="tweet",
		     id=dict_data["id"],
                     body={#"screen_name": dict_data["user"]["screen_name"],
                           # "@timestamp": dict_data["created_at"],
                           "id": dict_data["id"],
                           "timestamp": datetime.utcnow(),
                           "text": dict_data["text"],
                           "user": dict_data["user"],
                           "in_reply_to_user_id": dict_data["in_reply_to_user_id"],
                           "in_reply_to_user_id_str": dict_data["in_reply_to_user_id_str"],
                           "in_reply_to_screen_name": dict_data["in_reply_to_screen_name"],
                           "in_reply_to_status_id": dict_data["in_reply_to_status_id"],
                           "in_reply_to_status_id_str": dict_data["in_reply_to_status_id_str"],
                           "lang": dict_data["lang"],
                           "entities": dict_data["entities"],
                           "coordinates": dict_data["coordinates"],
                           "contributors": dict_data["contributors"],
                           "place": dict_data["place"],
                           # "quoted_status": dict_data["quoted_status"],
                           "retweet_count": dict_data["retweet_count"],
                           "retweeted": dict_data["retweeted"],
                           # "retweeted_status": dict_data["retweeted_status"],
                           # "possibly_sensitive": dict_data["possibly_sensitive"],
                           # "is_quote_status": dict_data["is_quote_status"],
                           "favorited": dict_data["favorited"],
                           "favorite_count": dict_data["favorite_count"],
                           "filter_level": dict_data["filter_level"],
                           "source": dict_data["source"],
                           "polarity": tweet.sentiment.polarity,
                           "subjectivity": tweet.sentiment.subjectivity,
                           "vader": vader,
                           "sentiment": sentiment})
            return True
        except:
            print('ERROR could not load tweet')
            return True

    # on failure
    def on_error(self, status):
        print status

if __name__ == '__main__':

    while True:
    	try:
            # create instance of the tweepy tweet stream listener
            listener = TweetStreamListener()

            # set twitter keys/tokens
            auth = OAuthHandler(config["consumer_key"], config["consumer_secret"])
            auth.set_access_token(config["access_token"], config["access_token_secret"])

            # create instance of the tweepy stream
            stream = Stream(auth, listener)

            stream.filter(follow=['3444040513', '1944611516', '1886806878', '1400086176', '862135092', '516599568', '512577026', '476266843', '465333646', '444987701', '431082451', '397517970', '389107198', '376983004', '371271630', '369723148', '349247026', '331689061', '282687344', '251655721', '250827256', '245129286', '239641139', '214650466', '212814156', '209716158', '209263514', '203157684', '202344988', '200901496', '198980530', '194715074', '194338802', '193843928', '193710578', '188080517', '183844516', '181273840', '169540701', '168741670', '166764258', '163203533', '159541526', '156723527', '141439375', '138771648', '135267043', '135252879', '133613215', '126044333', '126038022', '125360886', '124341360', '121843145', '115556805', '110512694', '110320384', '105891984', '103111232', '98566934', '94432907', '89505772', '87696613', '87229930', '85375262', '84642278', '83005870', '82117778', '81945782', '80610121', '79806078', '79512108', '78811893', '78506305', '78192935', '77951602', '77902742', '77577780', '77529140', '76473159', '76102930', '75033472', '73406718', '71344809', '71079334', '70126236', '69020299', '62934861', '62587361', '61524775', '59512284', '58821436', '58605733', '57710919', '56767252', '56004027', '52832338', '52513635', '52070893', '51189656', '51189120', '50323173', '50321644', '49880043', '49434877', '48760475', '48488561', '48224124', '47713256', '47243043', '46795961', '46791399', '46761152', '46682248', '45939528', '45645269', '44678264', '44507891', '44502120', '43010657', '42954457', '41229855', '39985597', '39616607', '39346451', '38721656', '38370519', '38251431', '38241499', '38032945', '36913910', '36434071', '35974550', '34745516', '34081596', '34013662', '32534362', '31782245', '31065027', '30590889', '30276184', '30187869', '30074516', '29780904', '29498970', '29203271', '29068597', '28592445', '27781943', '27710012', '27707996', '27649623', '26500119', '26322490', '26270913', '26074296', '26042217', '25932157', '25130217', '25067343', '24925573', '24903350', '24591063', '24486328', '24400732', '23803265', '23546948', '22537285', '22371792', '22185437', '21991753', '21923455', '21865953', '21833263', '21661825', '21487022', '21308488', '21112249', '21099064', '20950304', '20950240', '20818056', '20346956', '20265254', '20196159', '19792725', '19626505', '19564719', '19537303', '19418497', '19409270', '19361679', '19319374', '19263978', '19086513', '19077044', '18552281', '18481113', '18371803', '18360370', '18302609', '18263824', '18139461', '18004919', '17958426', '17931627', '17842064', '17518634', '17494046', '17345647', '17345146', '17292143', '17106279', '16794949', '16727749', '16440037', '16212685', '16201775', '15900167', '15817433', '14992591', '14608893', '14331674', '14328072', '11026952', '46653066', '7117962', '6395222', '667563' ], stall_warnings=True)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            print('ERROR!!!!!!!!!!!!!')
            sleep(30)
            pass





