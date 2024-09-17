from pathlib import Path
import os

SYSCAPS_PATH = Path( os.environ.get('SYSCAPS', ''))

captions_dir = SYSCAPS_PATH / 'captions' / 'resstock'

newlines_file = 'resstock_newlines.txt'
bad_caps_file = ['resstock_bad_caps_8.txt', 'resstock_bad_caps_9.txt', 'resstock_bad_caps_10.txt']

"""
with open(newlines_file, 'r') as f:
    files = f.readlines()

    for ff in files:
        ff=ff.strip()
        with open(captions_dir / 'long' / (ff+'.txt'), 'r') as capfile:
            cap = capfile.read().strip() # check this removes trailing newlines
        with open(captions_dir / 'long' / (ff+'.txt'), 'w') as capfile:
            capfile.write(cap)
"""
for bdf in bad_caps_file:
    with open(bdf, 'r') as f:
        files = f.readlines()

        for ff in files:
            ff = ff.strip()
            try:
                os.rename(captions_dir / 'long' / (ff+'.txt'), captions_dir / 'long_bad' / (ff+'.txt'))
            except:
                continue
