import sys

def edit_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_

    return distances[-1]

if __name__ == '__main__':
    file = sys.argv[1]
    with open(file, 'r') as f:
        lines = f.readlines()
        lines = [l.strip() for l in lines if l.strip() != '']
        pred = [lines[i] for i in range(0, len(lines), 2)]
        gold = [lines[i] for i in range(1, len(lines), 2)]
        pred_sent = [p.split()[3:] for p in pred]
        gold_sent = [g.split()[3:] for g in gold]
        c_ed, c_cnt = 0, 0
        w_ed, w_cnt = 0, 0
        for p, g in zip(pred_sent, gold_sent):
            print(' '.join(p))
            print(' '.join(g))
            s1 = list(' '.join(p))
            s2 = list(' '.join(g))
            c_ed += edit_distance(s1, s2)
            c_cnt += len(s2)
            w_ed += edit_distance(p, g)
            w_cnt += len(g)
        print('CER', c_ed/float(c_cnt))
        print('WER', w_ed/float(w_cnt))