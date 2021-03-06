import requests
import bs4
import re
import datetime
import time
import os

# should probably throw "orphan" pieces in a special bin, rather than tossing

sources = [ # think these are probably mostly piano
    # "https://kern.humdrum.org/cgi-bin/browse?l=/users/craig/classical", # metalist
    # "http://kern.humdrum.org/search?s=t&keyword=piano&type=Text",
    "http://kern.humdrum.org/search?s=t&keyword=waltz",
    "http://kern.humdrum.org/search?s=t&keyword=sonatina",
    "http://kern.humdrum.org/search?s=t&keyword=sonata",
    "http://kern.humdrum.org/search?s=t&keyword=scherzo",
    "http://kern.humdrum.org/search?s=t&keyword=prelude",
    "http://kern.humdrum.org/search?s=t&keyword=mazurka"
]

def formatted(href):
    """Return True iff the href appears to point to a **kern file. """
    # print(link)
    return href and ('.krn&format=kern' in href)

class Piece():
    """Allows me to write the functions monadically, without worrying
    about explicitly returning everything. """
    def __init__(self, url):
        self.url = url
        self.err = None

def is_piano(url):
    """Visit the webpage at `url`. Confirm that it's well-formatted **kern
    and contains a composer and title, and that it appears to be a piano
    piece. Return the corresponding Piece object, with an error at self.err
    if there were any errors. """
    # Extract the text at `url`
    piece = Piece(url)
    try:
        resp = requests.get(url)
        filetext = resp.text
        piece.raw = filetext
    except:
        piece.err = "failed to reach url"
        return piece
    # Check if text appears to contains **kern syntax 
    if filetext.strip() == "" or filetext.count("**kern") == 0:
        piece.err = "raw text doesn't appear well-formatted"
        return piece
    # Use regexs to find title, composer
    metadata = filetext[:filetext.find("*")]
    piece.metadata = metadata
    composer = re.search(r'!!!CO[MA]: ([^\n]*)\n', metadata)
    if composer is None:
        piece.err = "couldn't find composer"
        return piece
    piece.composer = composer
    title = re.search(r'!!!OTL: ([^\n]*)\n', metadata)
    if title is None:
        piece.err = "couldn't find title"
        return piece
    piece.title = title
    descriptor = composer.group(1)+"___"+title.group(1)
    piece.descriptor = descriptor # name for file to be generated
    # Look at the part of the file containing music, and check
    #   whether it seems to be piano music
    bars_and_notes = filetext[filetext.find("*"):]
    bars = bars_and_notes[:bars_and_notes.find("\n"):]
    m = re.fullmatch(r'\n?\s*\*\*kern\s*(\*\*dynam)?\s*\*\*kern\s*(\*\*dynam)?\s*', bars)
    if m is None:
        piece.err = "couldn't find the 2 piano bars"
        return piece
    return piece

def log_issues(text):
    """Create a log file (a new one every day) and log the issue `text` there. """
    log = datetime.date.today().strftime("%d%m%Y") #+ "_" + str(time.time())
    with open("log_"+log+".txt", "a") as file:
        file.write(text+"\n")

def download_all_piano_files(url):
    """Download all the piano **kern files at `url`. For each endpoint,
    download the file if possible, and log the error to a log file otherwise. """
    headers = {'Accept-Encoding': 'identity'}
    resp = requests.get(url, headers=headers)
    soup = bs4.BeautifulSoup(resp.text, features="lxml")
    links = soup.find_all(href=formatted)
    links = [link['href'] for link in links]
    for l in links:
        piece = is_piano(l)
        if piece.err is not None:
            log_issues(str(time.time()) + " " + piece.err + ": " + piece.url)
            # print(piece.err + ": " + piece.url)
        else:
            write_file(piece)


def write_file(piece: Piece):
    """Given the Piece object `piece`, write the piece's raw text to
    a new text file in the directory `raw`. """
    dir = 'raw'
    if not os.path.exists(dir):
        os.mkdir(dir)
    with open(dir + '/' + piece.descriptor, 'w') as file:
        file.write(piece.raw)


for source in sources:
    print('on '+source)
    download_all_piano_files(source)
    # write_files(to_write)

# is_piano('https://kern.humdrum.org/cgi-bin/ksdata?location=users/cutler/tmte/ch24&file=N6e.krn&format=kern')