import unicodedata
import re

import nltk

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords


DOMAIN_TYPE_MAPPING = dict()

## SQA
DOMAIN_TYPE_MAPPING['music'] = {'MusicRecording': 'Q7366', 'Person': 'Q5', 'MusicAlbum': 'Q208569', 'inLanguage': 'Q315'}   # Q5:human, Q7366:song, Q208569:studio_album

# TODO actor and director should be handled differently
DOMAIN_TYPE_MAPPING['movies'] = {'Movie': 'Q11424', 'Person.creator': 'Q2500638', 'Person.director': 'Q3455803', 'Person.actor': 'Q33999'}   # Q11424:film

# isbn is ignored for books because (1.bootleg don't detect them. 2.they are easy to recognize by semantic parser)
DOMAIN_TYPE_MAPPING['books'] = {'Book': 'Q571', 'Person': 'Q5', 'inLanguage': 'Q315', 'iso_lang_code': 'Q315', 'award': 'Q618779', 'bookEdition': 'Q57933693'}  # Q571:book, Q315:language, Q618779:award

DOMAIN_TYPE_MAPPING['linkedin'] = {'Organization': 'Q43229','Person': 'Q5', 'addressLocality': 'Q2221906', 'award': 'Q618779'} # Q2221906:geographic_location

DOMAIN_TYPE_MAPPING['restaurants'] = {'Restaurant': 'Q571', 'Person': 'Q5', 'servesCuisine': 'Q1778821', 'Location': 'Q2221906',
                                      'postalCode': 'Q37447', 'ratingValue': 'Q2283373', 'reviewCount': 'Q265158'}   # Q2283373:restaurant_rating, Q265158:review, Q1778821:cuisine

DOMAIN_TYPE_MAPPING['hotels'] = {'Hotel': 'Q571', 'LocationFeatureSpecification': 'Q5912147', 'Location': 'Q2221906',
                                 'CheckinTime': 'Q1068755', 'CheckoutTime': 'Q56353377', 'ratingValue': 'Q2976556'}   # Q5912147:hotel_amenity, Q2976556:hotel_rating


## Dialogues
DOMAIN_TYPE_MAPPING['spotify'] = {'song': 'Q7366', 'artist': 'Q5', 'artists': 'Q5', 'album': 'Q208569', 'genres': 'Q188451'}   # Q188451:music genre


BANNED_PHRASES = set(
    stopwords.words('english') + \
    ['music', 'musics', 'name', 'names', 'want', 'wants', 'album', 'albums', 'please', 'who', 'show me', 'tell me', 'find me', 'sing', 'sang',
     'play', 'play me', 'plays', 'track', 'tracks', 'song', 'songs', 'record', 'records', 'recordings', 'album', 'url', 'mount to',
     'something', 'get', 'selections', 'pages', 'isbn', 'isbn numbers', 'average rating', 'count', 'yesterday', 'before today', 'i need to know',
     'resume', 'resumes', 'the', 'search for me', 'search', 'searches', 'yes', 'yeah', 'popular', 'trouble', 'go', 'millisecond', 'good music', 'hear music',
     'h', 'm', 's', 'd', 'y', 'am', 'pm', 'min', 'sec', 'hour', 'year', 'month', 'day', 'us', 'we', 'who', 'what', 'where', 'the',
     'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
     'release', 'released', 'dance', 'dancing', 'need', 'i need', 'i would', ' i will', 'find', 'the list', 'get some', 'af', '1st', '2nd', '3rd',
     'tongue', 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'spotify', 'app', 'almond', 'genre',
     'play dj', 'stone', 'sound tracks', 'hi', 'hey', 'tweet', 'all music', 'hello', 'preference', 'top tracks', 'all the good', 'music i', 'id',
     'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec', 'from yesterday', 'show tunes', 'tag', 'ms', 'all good',
     'greatest hits', 'good hits']
)

BANNED_REGEX = [re.compile(r'\d (star|rating)'), re.compile(r'\dth'), re.compile(r'a \d'),
                re.compile(r'\d (hour|min|sec|minute|second|day|month|year)s?'), re.compile(r'this (hour|min|sec|minute|second|day|month|year)s?')]

def is_banned(word):
    return word in BANNED_PHRASES or any([regex.match(word) for regex in BANNED_REGEX])

def normalize_text(text):
    text = unicodedata.normalize('NFD', text).lower()
    text = re.sub('\s\s+', ' ', text)
    return text

def has_overlap(start, end, used_aliases):
    for alias in used_aliases:
        alias_start, alias_end = alias[1], alias[2]
        if start < alias_end and end > alias_start:
            return True
    return False


def post_process_bootleg_types(qid, type, title, almond_domains):
    # TODO if training on multiple domains (in one run) these mapping should be modified
    # e.g. song is mapped to book which is not correct if training on music domain too
    for domain in almond_domains:
        if domain == 'books':
            if type == 'Q15087423':
                type = 'unk'
    
            # [houghton mifflin award, ciudad de buenos aires award, new berry award]
            if qid in ['Q390074', 'Q1486', 'Q616527']:
                type = 'Q618779'
    
            # [penguin classics, ]
            elif qid in ['Q1336200']:
                type = 'Q57933693'
    
            elif 'book' in title or 'novel' in title or 'poem' in title or title in \
                    ['written work', 'literary work', 'literature', 'play', 'film', 'occurrence', 'song',
                     'fictional human', 'profession',
                     'document', 'day of the week', 'compilation album', 'magazine', 'television series', 'taxon',
                     'Bible translation',
                     'concept', 'disease', 'technique', 'activity', 'food', 'political ideology', 'literary genre',
                     'mountain', 'mental process',
                     'academic discipline', 'base material', 'negative emotion', 'emotion']:
                type = 'Q571'
            elif 'publisher' in title or title in ['editorial collection', 'version, edition, or translation']:
                type = 'Q57933693'
            elif 'person' in title or 'rights activist' in title or title in ['writer', 'journalist', 'author',
                                                                              'politician',
                                                                              'Esperantist', 'philosopher', 'actor',
                                                                              'painter',
                                                                              'historian', 'lawyer', 'poet', 'singer']:
                type = 'Q5'
            elif title in ['recurring event'] or 'award' in title:
                type = 'Q618779'
            # languages are not in typeid2title of bootleg
            # [language, country, ethnic group, people, republic]
            elif type in ['Q34770', 'Q6256', 'Q41710', 'Q2472587', 'Q7270']:
                type = 'Q315'
            elif title in ['day', 'single', 'musical group', 'English unit of measurement',
                           'Wikimedia disambiguation page', 'Wikimedia list article']:
                type = 'unk'
            else:
                type = 'unk'
            
        elif domain == 'spotify':
            # rap, rap music
            if qid in ['Q6010', 'Q11401']:
                type = 'Q188451'
    
            if title in ['song', 'single', 'musical composition', 'ballad', 'extended play', 'literary work',
                         'television series', 'film', 'play']:
                type = 'Q7366'
            elif 'album' in title or title in []:
                type = 'Q208569'
            elif 'genre' in title or title in ['country', 'music by country or region', 'music term', 'republic',
                                               'ethnic group', 'music scene']:
                type = 'Q188451'
            elif 'person' in title or 'musician' in title or title in ['singer', 'actor', 'musician', 'songwriter',
                                                                       'composer',
                                                                       'singer-songwriter', 'musical group', 'drummer',
                                                                       'writer',
                                                                       'poet', 'guitarist', 'rapper', 'painter',
                                                                       'film director',
                                                                       'rock band', 'university teacher', 'journalist',
                                                                       'television presenter',
                                                                       'saxophonist', 'music pedagogue',
                                                                       'association football player',
                                                                       'disc jockey', 'record producer', 'engineer',
                                                                       'human biblical figure', 'big band',
                                                                       'musical duo', 'girl group',
                                                                       'boy band', 'musical ensemble', 'artist',
                                                                       'vocal group', 'heavy metal band',
                                                                       'literary character', 'lawyer', 'lyricist',
                                                                       'baseball player']:
                type = 'Q5'
    
            elif title in ['video game', 'disease', 'city of the United States', 'taxon',
                           'Wikimedia disambiguation page', 'Wikimedia list article']:
                type = 'unk'

    return type

