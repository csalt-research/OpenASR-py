import os
from tqdm import tqdm

root_dir = 'indian'
data_dir = '/home/data/MCV-v3/clips'
format_str = '{}-{}.txt'

max_size = 250*1000

size_fn = lambda path: os.stat(data_dir + '/' + path).st_size

for split in ['train', 'val', 'test']:
    src_file = root_dir + '/' + format_str.format('src', split)
    tgt_file = root_dir + '/' + format_str.format('tgt', split)
    pairs = list(zip(open(src_file, 'r').readlines(), open(tgt_file, 'r').readlines()))
    sizes = [size_fn(p[0].strip()) for p in pairs]
    filtered_pairs = [(sz, sr, t) for sz, (sr, t) in tqdm(zip(sizes, pairs)) if sz<max_size]
    sorted_pairs = sorted(filtered_pairs, key=lambda p: p[0])
    szs, filtered_src, filtered_tgt = zip(*sorted_pairs)
    open(src_file + '.filtered', 'w+').writelines(filtered_src)
    open(tgt_file + '.filtered', 'w+').writelines(filtered_tgt)
    
    print(sorted(szs, reverse=True)[:10])
