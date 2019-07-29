import sentencepiece as spm

root_dir = 'us-v3'
vocab_size = 512

# train model
spm.SentencePieceTrainer.train('--input={0} --model_prefix={1} --vocab_size={2} --character_coverage=1'.format(root_dir + '/tgt-train.txt', root_dir, str(vocab_size)))
# decode sentences
sp = spm.SentencePieceProcessor()
sp.Load(root_dir + '.model')

for split in ['train', 'val', 'test']:
    inp = open(root_dir + '/tgt-' + split + '.txt', 'r')
    out = open(root_dir + '/tgt-' + split + '.txt.converted', 'w')
    lines = [t.strip() for t in inp.readlines()]
    conv_lines = [' '.join(sp.EncodeAsPieces(t)) for t in lines]
    out.writelines([t+'\n' for t in conv_lines])
    inp.close()
    out.close()
