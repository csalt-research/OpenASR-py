
def readlines(filepath):
	with open(filepath, 'r') as f:
		lst = f.readlines()
		lst = [l.strip() for l in lst]
	return lst

def writelines(filepath, lines):
	with open(filepath, 'w') as f:
		lst = [l+'\n' for l in lines]
		f.writelines(lst)

# utt2spk = 'utt2spk'
# spk2utt = 'spk2utt'
wav = 'wav.scp'
text = 'text'

utt_trans = readlines(text)
utt_trans = [(x.split()[0], ' '.join(x.split()[1:])) for x in utt_trans]
utt_id, trans = zip(*utt_trans)
writelines('tgt.txt', trans)

utt_wav = readlines(wav)
utt2wav = dict([x.split() for x in utt_wav])
paths = [utt2wav[u] for u in utt_id]
writelines('src.txt', paths)