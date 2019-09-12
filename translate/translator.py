import codecs
import time

import torch
from translate.beam_search import BeamSearch
from translate.translation import TranslationBuilder
from utils.misc import tile
from utils.logging import logger

class Translator(object):
    def __init__(
            self,
            model,
            dataloader,
            vocab,
            device,
            n_best=1,
            min_length=0,
            max_length=100,
            ratio=0.,
            beam_size=30,
            block_ngram_repeat=0,
            ignore_when_blocking=frozenset(),
            out_file=None,
            verbose=False,
            attn_debug=False):
        # model
        self.model = model

        # vocabulary
        self._tgt_vocab = vocab
        self._tgt_pad_idx = vocab.encode('<pad>')
        self._tgt_unk_idx = vocab.encode('<unk>')
        self._tgt_bos_idx = vocab.encode('<bos>')
        self._tgt_eos_idx = vocab.encode('<eos>')
        self._tgt_vocab_len = len(vocab)
        
        # device
        self._dev = device

        # decoding parameters
        self.n_best = n_best
        self.beam_size = beam_size
        self.min_length = min_length
        self.max_length = max_length
        self.ratio = ratio
        self.block_ngram_repeat = block_ngram_repeat
        self.ignore_when_blocking = ignore_when_blocking
        self._exclusion_idxs = {vocab.encode(t) for t in ignore_when_blocking}

        # dataloader
        self.dataloader = dataloader

        # logging
        self.out_file = open(out_file if out_file else '/dev/null', 'w+')
        self.verbose = verbose
        self.attn_debug = attn_debug

    def translate(self):
        xlation_builder = TranslationBuilder(self._tgt_vocab, self.n_best)

        # Statistics
        counter = 1
        pred_score_total, pred_words_total = 0, 0
        gold_score_total, gold_words_total = 0, 0

        all_scores = []
        all_predictions = []
        all_error_rates = []

        start_time = time.time()

        for batch in self.dataloader:
            src, src_len, tgt, tgt_len, feat = batch
            
            # Copy to device
            src = src.to(self._dev).contiguous()
            src_len = src_len.to(self._dev).contiguous()
            tgt = tgt.to(self._dev).contiguous()
            tgt_len = tgt_len.to(self._dev).contiguous()

            # Translate batch
            batch_data = self.translate_batch(
                src, src_len, tgt,
                max_length=self.max_length,
                min_length=self.min_length,
                ratio=self.ratio,
                n_best=self.n_best,
                return_attention=self.attn_debug
            )

            # Process translations
            translations = xlation_builder.from_batch(batch_data, tgt)
            for trans in translations:
                all_scores += [trans.pred_scores[:self.n_best]]
                pred_score_total += trans.pred_scores[0]
                pred_words_total += len(trans.pred_sents[0])
                if tgt is not None:
                    gold_score_total += trans.gold_score
                    gold_words_total += len(trans.gold_sent) + 1

                n_best_preds = [" ".join(pred)
                                for pred in trans.pred_sents[:self.n_best]]
                
                all_predictions += [n_best_preds]
                all_error_rates += [trans.error_rates()]

                self.out_file.write('\n'.join(n_best_preds) + '\n')
                self.out_file.flush()

                if self.verbose:
                    sent_number = counter
                    output = trans.log(sent_number)
                    logger.info(output)
                    counter += 1

                if self.attn_debug:
                    preds = trans.pred_sents[0]
                    preds.append('<eos>')
                    attns = trans.attns[0].tolist()
                    srcs = [str(item) for item in range(len(attns[0]))]
                    header_format = "{:>10.10} " + "{:>10.7} " * len(srcs)
                    row_format = "{:>10.10} " + "{:>10.7f} " * len(srcs)
                    output = header_format.format("", *srcs) + '\n'
                    for word, row in zip(preds, attns):
                        max_index = row.index(max(row))
                        row_format = row_format.replace(
                            "{:>10.7f} ", "{:*>10.7f} ", max_index + 1)
                        row_format = row_format.replace(
                            "{:*>10.7f} ", "{:>10.7f} ", max_index)
                        output += row_format.format(word, *row) + '\n'
                        row_format = "{:>10.10} " + "{:>10.7f} " * len(srcs)
                    logger.info(output)

        end_time = time.time()

        total_time = end_time - start_time
        logger.info("Total translation time (s): %f" % total_time)
        logger.info("Average translation time (s): %f" % (total_time / len(all_predictions)))
        logger.info("Tokens per second: %f" % (pred_words_total / total_time))

        return all_scores, all_predictions, all_error_rates

    def translate_batch(
            self,
            src, src_len, tgt,
            max_length,
            min_length=0,
            ratio=0.,
            n_best=1,
            return_attention=False):

        with torch.no_grad():
            # (0) Prep the components of the search.
            beam_size = self.beam_size
            batch_size = src.size(1)

            # (1) pt 1, Run the encoder on the src.
            enc_states, memory_bank, src_lengths = self.model.encoder(src, src_len)
            # (1) pt 2, Convert encoder state to decoder state
            enc_states = self.model.bridge(enc_states)
            # (1) pt 3, Make size of memory bank same as that of decoder
            memory_bank = self.model.W(memory_bank)
            self.model.decoder.init_state(enc_states)

            results = {
                "predictions": None,
                "scores": None,
                "attention": None,
                "gold_score": self._gold_score(tgt, enc_states, memory_bank, src_lengths) \
                    if tgt is not None else None
            }

            # (2) Repeat src objects `beam_size` times.
            # We use batch_size x beam_size
            self.model.decoder.map_state(
                lambda state, dim: tile(state, beam_size, dim=dim))

            memory_bank = tile(memory_bank, beam_size, dim=1)
            memory_lengths = tile(src_lengths, beam_size)

            # (0) pt 2, prep the beam object
            beam = BeamSearch(
                beam_size=beam_size,
                batch_size=batch_size,
                pad=self._tgt_pad_idx,
                bos=self._tgt_bos_idx,
                eos=self._tgt_eos_idx,
                n_best=n_best,
                device=self._dev,
                min_length=min_length,
                max_length=max_length,
                return_attention=return_attention,
                block_ngram_repeat=self.block_ngram_repeat,
                exclusion_tokens=self._exclusion_idxs,
                memory_lengths=memory_lengths,
                ratio=ratio
            )

            for step in range(max_length):
                decoder_input = beam.current_predictions.view(1, -1, 1)

                log_probs, attn = self.model.decoder(
                    decoder_input, memory_bank, memory_lengths=memory_lengths)
                log_probs = log_probs.squeeze(0)

                beam.advance(log_probs, attn)
                any_beam_is_finished = beam.is_finished.any()
                if any_beam_is_finished:
                    beam.update_finished()
                    if beam.done:
                        break

                select_indices = beam.current_origin

                if any_beam_is_finished:
                    # Reorder states.
                    memory_bank = memory_bank.index_select(1, select_indices)
                    memory_lengths = memory_lengths.index_select(0, select_indices)

                self.model.decoder.map_state(
                    lambda state, dim: state.index_select(dim, select_indices))

            results["scores"] = beam.scores
            results["predictions"] = beam.predictions
            results["attention"] = beam.attention
            return results

    def _gold_score(self, tgt, enc_states, memory_bank, src_lengths):
        # Decoder input
        tgt_in = tgt[:-1]
        # F-prop via decoder
        log_probs, _ = self.model.decoder(
            tgt_in, memory_bank, memory_lengths=src_lengths)
        log_probs = log_probs.squeeze(0)
        # Ignore log-probs for padding
        log_probs[:, :, self._tgt_pad_idx] = 0
        gold = tgt[1:]
        gold_scores = log_probs.gather(2, gold)
        gold_scores = gold_scores.sum(dim=0).view(-1)
        # Revert back decoder state to original
        self.model.decoder.init_state(enc_states)
        return gold_scores