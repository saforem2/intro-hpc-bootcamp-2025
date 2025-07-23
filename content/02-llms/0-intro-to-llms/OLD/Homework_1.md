# Let’s write an elementary tokenizer that uses words as tokens.
Sam Foreman
2025-07-22

<link rel="preconnect" href="https://fonts.googleapis.com">

We will use Mark Twain’s *Life On The Mississippi* as a test bed. The
text is in the accompanying file ‘Life_On_The_Mississippi.txt’

Here’s a not-terribly-good such tokenizer:

``` python
wdict = {}
with open('Life_On_The_Mississippi.txt', 'r') as L:
    line = L.readline()
    nlines = 1
    while line:

        words = line.split()
        for word in words:
            if wdict.get(word) is not None:
                wdict[word] += 1
            else:
                wdict[word] = 1
        line = L.readline()
        nlines += 1

nitem = 0 ; maxitems = 100
for item in wdict.items():
    nitem += 1
    print(item)
    if nitem == maxitems: break
```

    ('\ufeffThe', 1)
    ('Project', 79)
    ('Gutenberg', 22)
    ('eBook', 4)
    ('of', 4469)
    ('Life', 5)
    ('on', 856)
    ('the', 8443)
    ('Mississippi', 104)
    ('This', 127)
    ('ebook', 2)
    ('is', 1076)
    ('for', 1017)
    ('use', 34)
    ('anyone', 4)
    ('anywhere', 8)
    ('in', 2381)
    ('United', 36)
    ('States', 26)
    ('and', 5692)
    ('most', 119)
    ('other', 223)
    ('parts', 5)
    ('world', 40)
    ('at', 676)
    ('no', 325)
    ('cost', 18)
    ('with', 1053)
    ('almost', 37)
    ('restrictions', 2)
    ('whatsoever.', 2)
    ('You', 92)
    ('may', 85)
    ('copy', 12)
    ('it,', 199)
    ('give', 67)
    ('it', 1382)
    ('away', 107)
    ('or', 561)
    ('re-use', 2)
    ('under', 112)
    ('terms', 22)
    ('License', 8)
    ('included', 2)
    ('this', 591)
    ('online', 4)
    ('www.gutenberg.org.', 4)
    ('If', 85)
    ('you', 813)
    ('are', 361)
    ('not', 680)
    ('located', 9)
    ('States,', 8)
    ('will', 287)
    ('have', 557)
    ('to', 3518)
    ('check', 4)
    ('laws', 13)
    ('country', 50)
    ('where', 152)
    ('before', 150)
    ('using', 10)
    ('eBook.', 2)
    ('Title:', 1)
    ('Author:', 1)
    ('Mark', 2)
    ('Twain', 2)
    ('Release', 1)
    ('date:', 1)
    ('July', 7)
    ('10,', 2)
    ('2004', 1)
    ('[eBook', 1)
    ('#245]', 1)
    ('Most', 4)
    ('recently', 3)
    ('updated:', 1)
    ('January', 2)
    ('1,', 2)
    ('2021', 1)
    ('Language:', 1)
    ('English', 7)
    ('Credits:', 1)
    ('Produced', 2)
    ('by', 623)
    ('David', 2)
    ('Widger.', 2)
    ('Earliest', 2)
    ('PG', 3)
    ('text', 4)
    ('edition', 3)
    ('produced', 15)
    ('Graham', 2)
    ('Allan', 2)
    ('***', 4)
    ('START', 1)
    ('OF', 16)
    ('THE', 29)
    ('PROJECT', 4)
    ('GUTENBERG', 3)

This is unsatisfactory for a few reasons:

- There are non-ASCII (Unicode) characters that should be stripped (the
  so-called “Byte-Order Mark” or BOM at the beginning of the text);

- There are punctuation marks, which we don’t want to concern ourselves
  with;

- The same word can appear capitalized, or lower-case, or with its
  initial letter upper-cased, whereas we want them all to be normalized
  to lower-case.

Part 1 of this assignment: insert code in this loop to operate on the
str variable ‘line’ so as to fix these problems before ‘line’ is split
into words.

A hint to one possible way to do this: use the ‘punctuation’ character
definition in the Python ‘string’ module, the ‘maketrans’ and
‘translate’ methods of Python’s str class, to eliminate punctuation, and
the regular expression (‘re’) Python module to eliminate any Unicode—it
is useful to know that the regular expression r’\[^\x00-x7f\]’ means
“any character not in the vanilla ASCII set.

Part 2: Add code to sort the contents of wdict by word occurrence
frequency. What are the top 100 most frequent word tokens? Adding up
occurrence frequencies starting from the most frequent words, how many
distinct words make up the top 90% of word occurrences in this “corpus”?

For this part, the docs of Python’s ‘sorted’ and of the helper
‘itemgetter’ from ‘operator’ reward study.

Write your modified code in the cell below.
