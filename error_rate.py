import os
import argparse
import Levenshtein as Lev

def edit_distance(l1, l2):
    b = set(l1 + l2)
    tok2char = dict(zip(b, range(len(b))))
    w1 = [chr(tok2char[t]) for t in l1]
    w2 = [chr(tok2char[t]) for t in l2]
    return Lev.distance(''.join(w1), ''.join(w2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', 
        type=str, help='Model predictions', required=True)
    parser.add_argument('--tgt', 
        type=str, help='Target predictions', required=True)
    parser.add_argument('--token', 
        type=str, help='Granularity of predictions', required=True, choices=['word', 'char', 'sp'])
    args = parser.parse_args()

    assert os.path.exists(args.pred)
    assert os.path.exists(args.tgt)

    f1 = open(args.pred, 'r')
    f2 = open(args.tgt, 'r')

    l1 = [x.strip().split() for x in f1.readlines()]
    l2 = [x.strip().split() for x in f2.readlines()]

    wer, cer = 0., 0.
    n_w, n_c = 0, 0
    if args.token == 'word':
        for x, y in zip(l1, l2):
            # WER
            d = edit_distance(x, y)
            wer += d
            n_w += len(y)
            # CER
            c_x = list('_'.join(x))
            c_y = list('_'.join(y))
            d = edit_distance(c_x, c_y)
            cer += d
            n_c += len(c_y)
        print('WER: %.2f CER: %.2f' % (100*wer/n_w, 100*cer/n_c))
    elif args.token == 'char':
        for x, y in zip(l1, l2):
            # CER
            d = edit_distance(x, y)
            cer += d
            n_c += len(y)
            # WER
            w_x = ''.join(x).split('_')
            w_y = ''.join(y).split('_')
            d = edit_distance(w_x, w_y)
            wer += d
            n_w += len(w_y)
        print('WER: %.2f CER: %.2f' % (100*wer/n_w, 100*cer/n_c))
    elif args.token == 'sp':
        for x, y in zip(l1, l2):
            # CER
            d = edit_distance(list(''.join(x).replace('▁', ' ').strip()), list(''.join(y).replace('▁', ' ').strip()))
            cer += d
            n_c += len(' '.join(''.join(y).replace('▁', ' ').split()))
            # WER
            w_x = ''.join(x).replace('▁', ' ').split()
            w_y = ''.join(y).replace('▁', ' ').split()
            d = edit_distance(w_x, w_y)
            wer += d
            n_w += len(w_y)
        print('WER: %.2f CER: %.2f' % (100*wer/n_w, 100*cer/n_c))
    else:
        print('Unsupported token type.')
