import os
from tqdm import tqdm

root_dir = 'us-v3'
data_dir = '/home/data/MCV-v3/clips'
format_str = '{}-{}.txt'

valid_fn = lambda path: os.stat(data_dir + '/' + path).st_size < 250*1000

for split in ['train', 'val', 'test']:
    src_file = root_dir + '/' + format_str.format('src', split)
    tgt_file = root_dir + '/' + format_str.format('tgt', split)
    pairs = zip(open(src_file, 'r').readlines(), open(tgt_file, 'r').readlines())
    filtered_pairs = [p for p in tqdm(pairs) if valid_fn(p[0].strip())]
    filtered_src, filtered_tgt = zip(*filtered_pairs)
    open(src_file + '.filtered', 'w+').writelines(filtered_src)
    open(tgt_file + '.filtered', 'w+').writelines(filtered_tgt)
    
    files = open(src_file, 'r').readlines()
    sz = [os.stat(data_dir + '/' + p.strip()).st_size for p in files]
    print(sorted(sz, reverse=True)[:10])
