import requests
import re
import numpy as np
import pandas as pd
from threading import Thread
from queue import Queue

# function to get html text
def get_html(url):
    html = requests.get(url)
    return html.text

# function to get relevant urls from the main page
def get_urls(main_url):
    page = get_html(main_url)
    all_urls = re.findall('<a href="(.*?)"', page, re.DOTALL)
    return all_urls

# function to get text from each plays url
def get_plays(url):
    page = get_html(url).lower()
    playsname = re.findall('<td class="play" align="center">(.*?)<', page, re.DOTALL)[0].strip() #name of plays
    act = int(url.rsplit('/', 1)[1].rsplit('.')[1]) #act
    scene = int(url.rsplit('/', 1)[1].rsplit('.')[2]) #scene
    title = re.findall('<title>(.*?)<', page, re.DOTALL)[0].strip() #title of work
    names = []
    speeches = []

    # get speeches
    speeches_region = re.split('speech1>', page)[1]

    # get each speech
    name_speeches = re.findall('<b>(.*?)</blockquote>', speeches_region, re.DOTALL) #name and speech
    for s in name_speeches:
        name = re.findall('(.*?)</b>', s, re.DOTALL)[0].strip() #get name
        raw_speech = re.split('<blockquote>', s)[1] #get all lines
        lines = re.findall('[0-9]>(.*?)</', raw_speech, re.DOTALL) #get list of lines
        speech = [line.strip() for line in lines]
        names.append(name)
        speeches.append(speech)
    print(len(names), len(speeches))

    # load to pandas df
    df = pd.DataFrame(columns=['playname', 'title', 'act', 'scene', 'line', 'speaker', 'speech'])
    for i in range(len(speeches)):
        df.loc[i] = [playsname, title, act, scene, i, names[i], speeches[i]]
    print(url + ' load complete')
    return df

# function to get text from each other plays url
def get_other_plays(url):
    page = get_html(url).lower()
    playsname = re.findall('<td class="play" align="center">(.*?)<', page, re.DOTALL)[0].strip() #name of plays
    act = int(url.rsplit('/', 1)[1].rsplit('.')[1]) #act
    scene = int(url.rsplit('/', 1)[1].rsplit('.')[2]) #scene
    title = re.findall('<title>(.*?)<', page, re.DOTALL)[0].strip() #title of work
    names = []
    speeches = []

    # get speeches ---FIX THIS
    speeches_region = re.split('speech1>', page)[1]

    # get each speech
    name_speeches = re.findall('<b>(.*?)</blockquote>', speeches_region, re.DOTALL) #name and speech
    for s in name_speeches:
        name = re.findall('(.*?)</b>', s, re.DOTALL)[0].strip() #get name
        raw_speech = re.split('<blockquote>', s)[1] #get all lines
        lines = re.findall('[0-9]>(.*?)</', raw_speech, re.DOTALL) #get list of lines
        speech = [line.strip() for line in lines]
        names.append(name)
        speeches.append(speech)
    print(len(names), len(speeches))

    # load to pandas df
    df = pd.DataFrame(columns=['playname', 'title', 'act', 'scene', 'line', 'speaker', 'speech'])
    for i in range(len(speeches)):
        df.loc[i] = [playsname, title, act, scene, i, names[i], speeches[i]]
    print(url + ' load complete')
    return df

# function to get text from each prologue url
def get_prologue(url):
    page = get_html(url).lower()
    playsname = re.findall('<td class="play" align="center">(.*?)<', page, re.DOTALL)[0].strip() #name of plays
    act = int(url.rsplit('/', 1)[1].rsplit('.')[1]) #act
    scene = int(url.rsplit('/', 1)[1].rsplit('.')[2]) #scene
    title = re.findall('<title>(.*?)<', page, re.DOTALL)[0].strip() #title of work
    speeches = []

    # get speeches
    speeches_region = re.split('</h3>', page)[1]

    # get each speech
    lines = re.findall('=[0-9+]>(.*?)</a', speeches_region, re.DOTALL) #get list of lines
    if lines:
        speech = [line.strip() for line in lines]
        speeches.append(speech)
    print(len(speeches))

    # load to pandas df
    df = pd.DataFrame(columns=['playname', 'title', 'act', 'scene', 'line', 'speaker', 'speech'])
    for i in range(len(speeches)):
        df.loc[i] = [playsname, title, act, scene, i, 'Prologue', speeches[i]]
    print(url + ' load complete')
    return df


# function to get text from each poetry url
def get_poetry(url):

    return


## get all urls to txt files
def save_urls():
    root = 'http://shakespeare.mit.edu'
    all_main_urls = get_urls(root)

    ## separate Poetry from Plays
    poetry_main_urls = [url for url in all_main_urls if (url[:6] == 'Poetry')]
    plays_main_urls = [url for url in all_main_urls if (url not in poetry_main_urls)]
    poetry_main_urls = [root + '/' + url for url in poetry_main_urls] #add prefix
    plays_main_urls = [root + '/' + url for url in plays_main_urls] #add prefix

    ## get urls inside each plays
    plays_all_urls = []
    prologue_all_urls = []
    plays_all_others_urls = []

    for url in plays_main_urls:
        prefix = re.findall('mit.edu/(.*?)/', url)[0]
        plays_urls = get_urls(url)
        plays_urls = [url for url in plays_urls if (url[-5:] == '.html' and url != 'full.html')]

        plays_only_urls = [url for url in plays_urls if (url[-7:] != '.0.html' and url[:11] != '3henryvi.5.')] #exclude prologue
        prologue_urls = [url for url in plays_urls if (url[-7:] == '.0.html')] #get prologue
        plays_other_urls = [url for url in plays_urls if (url[-7:] != '.0.html' and url[:11] == '3henryvi.5.')] #get 3henryvi.5.x

        plays_only_urls = [root + '/' + prefix + '/' + url for url in plays_only_urls] #add prefix
        prologue_urls = [root + '/' + prefix + '/' + url for url in prologue_urls] #add prefix
        plays_other_urls = [root + '/' + prefix + '/' + url for url in plays_other_urls] #add prefix
        plays_all_urls.extend(plays_only_urls) #add to list
        prologue_all_urls.extend(prologue_urls) #add to list
        plays_all_others_urls.extend(plays_other_urls) #add to list

    ## write to .txt
    with open('urls_plays.txt', 'w') as f:
        for x in plays_all_urls:
            f.write(x)
            f.write('\n')

    with open('urls_plays_other.txt', 'w') as f:
        for x in plays_all_others_urls:
            f.write(x)
            f.write('\n')

    with open('urls_prologue.txt', 'w') as f:
        for x in prologue_all_urls:
            f.write(x)
            f.write('\n')

    return


#### Main ####


## read urls from txt files
with open('urls_plays.txt') as f:
    plays = f.readlines()

with open('urls_plays_other.txt') as f:
    others = f.readlines()

with open('urls_prologue.txt') as f:
    prologues = f.readlines()

'''
## get texts from all plays in dataframe
frames = []
i = 0
for url in plays:
    frames.append(get_plays(url))
    i += 1
    print(i)
for url in others:
    frames.append(get_others(url))
    i += 1
    print(i)
for url in prologues:
    frames.append(get_prologue(url))
df = pd.concat(frames, ignore_index=True)
df.to_csv('allplays.txt', sep='\t') #write to tsv
'''

## doesn't work for 3henryvi.5.x
