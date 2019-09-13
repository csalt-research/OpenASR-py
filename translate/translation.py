import torch
from utils.misc import edit_distance

class TranslationBuilder(object):
    def __init__(self, vocab, n_best):
        self.vocab = vocab
        self.n_best = n_best

    def _build_target_tokens(self, pred, attn):
        tokens = []
        for tok in pred:
            tokens.append(self.vocab.decode(tok))
            if tokens[-1] == '<eos>':
                tokens = tokens[:-1]
                break
        return tokens

    def from_batch(self, translation_batch, tgt=None):
        batch_size = len(translation_batch["predictions"])
        preds, pred_score, attn, gold_score = \
            translation_batch["predictions"], \
            translation_batch["scores"], \
            translation_batch["attention"], \
            translation_batch["gold_score"]

        translations = []
        for b in range(batch_size):
            pred_sents = [self._build_target_tokens(preds[b][n], attn[b][n]) \
                          for n in range(self.n_best)]
            gold_sent = None
            if tgt is not None:
                gold_sent = self._build_target_tokens(tgt[1:, b], None)

            translation = Translation(
                pred_sents, attn[b], pred_score[b],
                gold_sent, gold_score[b]
            )
            translations.append(translation)

        return translations


class Translation(object):
    def __init__(self, pred_sents, attn, pred_scores, tgt_sent, gold_score):
        self.pred_sents = pred_sents
        self.attns = attn
        self.pred_scores = pred_scores
        self.gold_sent = tgt_sent
        self.gold_score = gold_score

    def log(self, sent_number):
        """ Log translation. """
        msg = []
        best_pred = self.pred_sents[0]
        best_score = self.pred_scores[0]
        pred_sent = ''.join(best_pred).replace('\u2581', ' ')
        msg.append('\nPRED {}: [{:.4f}] {}\n'.format(sent_number, best_score, pred_sent))

        if self.gold_sent is not None:
            tgt_sent = ''.join(self.gold_sent).replace('\u2581', ' ')
            msg.append('GOLD {}: [{:.4f}] {}\n'.format(sent_number, self.gold_score, tgt_sent))
        if len(self.pred_sents) > 1:
            msg.append('\nBEST HYP:\n')
            for score, sent in zip(self.pred_scores, self.pred_sents):
                msg.append("[{:.4f}] {}\n".format(score, sent))

        return "".join(msg)

    def error_rates(self):
        result = {}
        if self.gold_sent is not None:
            # wordpiece error rate
            s1 = self.gold_sent
            s2 = self.pred_sents[0]
            ed = edit_distance(s1, s2)
            cnt = len(s1)
            result['ER'] = (ed, cnt)
            # word error rate
            s1 = ''.join(self.gold_sent).split('\u2581')
            s2 = ''.join(self.pred_sents[0]).split('\u2581')
            ed = edit_distance(s1, s2)
            cnt = len(s1)
            result['WER'] = (ed, cnt)
            # character error rate
            s1 = list(''.join(self.gold_sent).replace('\u2581', ' '))
            s1 = list(''.join(self.pred_sents[0]).replace('\u2581', ' '))
            ed = edit_distance(s1, s2)
            cnt = len(s1)
            result['CER'] = (ed, cnt)
        else:
            result['ER'] = (0, 0)
            result['WER'] = (0, 0)
            result['CER'] = (0, 0)
        return result