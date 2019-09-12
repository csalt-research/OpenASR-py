import torch

class TranslationBuilder(object):
    def __init__(self, vocab, n_best):
        self.vocab = vocab
        self.n_best = n_best

    def _build_target_tokens(self, pred, attn):
        EOS_id = self.vocab.encode('<eos>')
        tokens = []
        for tok in pred:
            tokens.append(self.vocab.encode(tok))
            if tokens[-1] == EOS_id:
                tokens = tokens[:-1]
                break
        return tokens

    def from_batch(self, translation_batch, tgt=None):
        batch_size = len(translation_batch["predictions"])
        preds, pred_score, attn, gold_score = \
            translation_batch["predictions"],
            translation_batch["scores"],
            translation_batch["attention"],
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
        pred_sent = ' '.join(best_pred)
        msg.append('PRED {}: {}\n'.format(sent_number, pred_sent))
        msg.append("PRED SCORE: {:.4f}\n".format(best_score))

        if self.gold_sent is not None:
            tgt_sent = ' '.join(self.gold_sent)
            msg.append('GOLD {}: {}\n'.format(sent_number, tgt_sent))
            msg.append(("GOLD SCORE: {:.4f}\n".format(self.gold_score)))
        if len(self.pred_sents) > 1:
            msg.append('\nBEST HYP:\n')
            for score, sent in zip(self.pred_scores, self.pred_sents):
                msg.append("[{:.4f}] {}\n".format(score, sent))

        return "".join(msg)
