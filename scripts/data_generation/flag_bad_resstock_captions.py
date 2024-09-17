from nltk import ngrams
import os
from pathlib import Path 
from tqdm import tqdm 

SYSCAPS_PATH = Path( os.environ.get('SYSCAPS', '') )

ns = [8,9,10]
captions_dir = SYSCAPS_PATH / 'captions' / 'resstock' / 'long'
caps = captions_dir.glob('*')
caps = [x for x in caps if x.is_file()]

whitefile = open('resstock_newlines_2.txt', 'w')

punc = ['.', ',', '!', ':', ';', '*']

for c in tqdm(caps):
    with open(c, 'r') as f:
        cap = f.read().lower()
        if cap[-3:] == '\n'*3:
            whitefile.write(str(c.stem) + '\n')
        else:
            cap = cap.split()
            cap = [c for c in cap if c not in punc]
            for n in ns:
                dupfile = open(f'resstock_bad_caps_{n}.txt', 'a+')
                grams = list(ngrams(cap, n))
                grams = grams[::n]
                grams = [','.join(g) for g in grams]
                if len(grams) >= 5+len(set(grams)):
                    dupfile.write(str(c.stem) + '\n')
                dupfile.close()
whitefile.close()
