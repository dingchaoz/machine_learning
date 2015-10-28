import pandas as pd
import re
from nltk.tokenize import wordpunct_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

## tokenize, stem, remove stopwords
def tokenStem(words):
    words = words.strip('[').strip(']').lower() #remove brackets and lowercase
    words = re.sub('[(){}<>:,.!?\'"]', '', words)
    stemmer = PorterStemmer()
    stops = stopwords.words('english')
    output = [stemmer.stem(token) for token in wordpunct_tokenize(words) if token not in stops ] #stem words
    return " ".join(output) #merge into strings

#### Main ####
if __name__ == '__main__':

    # import file
    df = pd.read_csv('all_poems.txt', sep='\t')

    # tokenize, stem, and remove stopwords
    df['poem'] = df['poem'].map(lambda x: tokenStem(x))

    # write to 'all_plays_tokenized.txt'
    df.to_csv('all_poems_tokenized.txt', sep='\t', index=False)

