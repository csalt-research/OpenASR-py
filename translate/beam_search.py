import torch

class BeamSearch(object):
    def __init__(self, 
                 beam_size, 
                 batch_size, 
                 pad, 
                 bos, 
                 eos, 
                 n_best, 
                 device,
                 min_length, 
                 max_length, 
                 return_attention,
                 block_ngram_repeat, 
                 exclusion_tokens, 
                 memory_lengths,
                 ratio):
        super(BeamSearch, self).__init__()
        # magic indices
        self.pad = pad
        self.bos = bos
        self.eos = eos

        # result caching
        self.predictions = [[] for _ in range(batch_size)]
        self.scores = [[] for _ in range(batch_size)]
        self.attention = [[] for _ in range(batch_size)]

        self.alive_seq = torch.full(
            [batch_size * beam_size, 1], self.bos,
            dtype=torch.long, device=device)
        self.is_finished = torch.zeros(
            [batch_size, beam_size],
            dtype=torch.uint8, device=device)
        self.alive_attn = None

        self.min_length = min_length
        self.max_length = max_length
        self.block_ngram_repeat = block_ngram_repeat
        self.exclusion_tokens = exclusion_tokens
        self.return_attention = return_attention

        # beam parameters
        # self.global_scorer = global_scorer
        self.beam_size = beam_size
        self.n_best = n_best
        self.batch_size = batch_size
        self.ratio = ratio

        # result caching
        self.hypotheses = [[] for _ in range(batch_size)]

        # beam state
        self.top_beam_finished = torch.zeros([batch_size], dtype=torch.uint8)
        self.best_scores = torch.full([batch_size], -1e10, dtype=torch.float,
                                      device=device)

        self._batch_offset = torch.arange(batch_size, dtype=torch.long)
        self._beam_offset = torch.arange(
            0, batch_size * beam_size, step=beam_size, dtype=torch.long,
            device=device)
        self.topk_log_probs = torch.tensor(
            [0.0] + [float("-inf")] * (beam_size - 1), device=device
        ).repeat(batch_size)
        self.select_indices = None
        self._memory_lengths = memory_lengths

        # buffers for the topk scores and 'backpointer'
        self.topk_scores = torch.empty((batch_size, beam_size),
                                       dtype=torch.float, device=device)
        self.topk_ids = torch.empty((batch_size, beam_size), dtype=torch.long,
                                    device=device)
        self._batch_index = torch.empty([batch_size, beam_size],
                                        dtype=torch.long, device=device)
        self.done = False

    @property
    def current_predictions(self):
        return self.alive_seq[:, -1]

    @property
    def current_origin(self):
        return self.select_indices

    def __len__(self):
        return self.alive_seq.shape[1]

    def ensure_min_length(self, log_probs):
        if len(self) <= self.min_length:
            log_probs[:, self.eos] = -1e20

    def ensure_max_length(self):
        # add one to account for BOS. Don't account for EOS because hitting
        # this implies it hasn't been found.
        if len(self) == self.max_length + 1:
            self.is_finished.fill_(1)

    def block_ngram_repeats(self, log_probs):
        cur_len = len(self)
        if self.block_ngram_repeat > 0 and cur_len > 1:
            for path_idx in range(self.alive_seq.shape[0]):
                # skip BOS
                hyp = self.alive_seq[path_idx, 1:]
                ngrams = set()
                fail = False
                gram = []
                for i in range(cur_len - 1):
                    # Last n tokens, n = block_ngram_repeat
                    gram = (gram + [hyp[i].item()])[-self.block_ngram_repeat:]
                    # skip the blocking if any token in gram is excluded
                    if set(gram) & self.exclusion_tokens:
                        continue
                    if tuple(gram) in ngrams:
                        fail = True
                    ngrams.add(tuple(gram))
                if fail:
                    log_probs[path_idx] = -10e20

    def advance(self, log_probs, attn):
        vocab_size = log_probs.size(-1)

        # using integer division to get an integer _B without casting
        _B = log_probs.shape[0] // self.beam_size

        # force the output to be longer than self.min_length
        step = len(self)
        self.ensure_min_length(log_probs)

        # Multiply probs by the beam probability.
        log_probs += self.topk_log_probs.view(_B * self.beam_size, 1)

        self.block_ngram_repeats(log_probs)

        # if the sequence ends now, then the penalty is the current
        # length + 1, to include the EOS token
        length_penalty = 1. # not applying length penalty cufrently

        # Flatten probs into a list of possibilities.
        curr_scores = log_probs / length_penalty
        curr_scores = curr_scores.reshape(_B, self.beam_size * vocab_size)
        torch.topk(curr_scores,  self.beam_size, dim=-1,
                   out=(self.topk_scores, self.topk_ids))

        # Recover log probs.
        # Length penalty is just a scalar. It doesn't matter if it's applied
        # before or after the topk.
        torch.mul(self.topk_scores, length_penalty, out=self.topk_log_probs)

        # Resolve beam origin and map to batch index flat representation.
        torch.div(self.topk_ids, vocab_size, out=self._batch_index)
        self._batch_index += self._beam_offset[:_B].unsqueeze(1)
        self.select_indices = self._batch_index.view(_B * self.beam_size)

        self.topk_ids.fmod_(vocab_size)  # resolve true word ids

        # Append last prediction.
        self.alive_seq = torch.cat(
            [self.alive_seq.index_select(0, self.select_indices),
             self.topk_ids.view(_B * self.beam_size, 1)], -1)

        if self.return_attention:
            current_attn = attn.index_select(1, self.select_indices)
            if step == 1:
                self.alive_attn = current_attn
                # update global state (step == 1)
            else:
                self.alive_attn = self.alive_attn.index_select(
                    1, self.select_indices)
                self.alive_attn = torch.cat([self.alive_attn, current_attn], 0)

        self.is_finished = self.topk_ids.eq(self.eos)
        self.ensure_max_length()

    def update_finished(self):
        # Penalize beams that finished.
        _B_old = self.topk_log_probs.shape[0]
        step = self.alive_seq.shape[-1]  # 1 greater than the step in advance
        self.topk_log_probs.masked_fill_(self.is_finished, -1e10)
        self.is_finished = self.is_finished.to('cpu')
        self.top_beam_finished |= self.is_finished[:, 0].eq(1)
        predictions = self.alive_seq.view(_B_old, self.beam_size, step)
        attention = (
            self.alive_attn.view(
                step - 1, _B_old, self.beam_size, self.alive_attn.size(-1))
            if self.alive_attn is not None else None)
        non_finished_batch = []
        for i in range(self.is_finished.size(0)):
            b = self._batch_offset[i]
            finished_hyp = self.is_finished[i].nonzero().view(-1)
            # Store finished hypotheses for this batch.
            for j in finished_hyp:
                if self.ratio > 0:
                    s = self.topk_scores[i, j] / (step + 1)
                    if self.best_scores[b] < s:
                        self.best_scores[b] = s
                self.hypotheses[b].append((
                    self.topk_scores[i, j],
                    predictions[i, j, 1:],  # Ignore start_token.
                    attention[:, i, j, :self._memory_lengths[i]]
                    if attention is not None else None))
            # End condition is the top beam finished and we can return
            # n_best hypotheses.
            if self.ratio > 0:
                pred_len = self._memory_lengths[i] * self.ratio
                finish_flag = ((self.topk_scores[i, 0] / pred_len)
                               <= self.best_scores[b]) or \
                    self.is_finished[i].all()
            else:
                finish_flag = self.top_beam_finished[i] != 0
            if finish_flag and len(self.hypotheses[b]) >= self.n_best:
                best_hyp = sorted(
                    self.hypotheses[b], key=lambda x: x[0], reverse=True)
                for n, (score, pred, attn) in enumerate(best_hyp):
                    if n >= self.n_best:
                        break
                    self.scores[b].append(score)
                    self.predictions[b].append(pred)
                    self.attention[b].append(
                        attn if attn is not None else [])
            else:
                non_finished_batch.append(i)
        non_finished = torch.tensor(non_finished_batch)
        # If all sentences are translated, no need to go further.
        if len(non_finished) == 0:
            self.done = True
            return

        # Remove finished batches for the next step.
        _B_new = non_finished.shape[0]
        self.top_beam_finished = self.top_beam_finished.index_select(
            0, non_finished)
        self._batch_offset = self._batch_offset.index_select(0, non_finished)
        non_finished = non_finished.to(self.topk_ids.device)
        self.topk_log_probs = self.topk_log_probs.index_select(0,
                                                               non_finished)
        self._batch_index = self._batch_index.index_select(0, non_finished)
        self.select_indices = self._batch_index.view(_B_new * self.beam_size)
        self.alive_seq = predictions.index_select(0, non_finished) \
            .view(-1, self.alive_seq.size(-1))
        self.topk_scores = self.topk_scores.index_select(0, non_finished)
        self.topk_ids = self.topk_ids.index_select(0, non_finished)
        if self.alive_attn is not None:
            inp_seq_len = self.alive_attn.size(-1)
            self.alive_attn = attention.index_select(1, non_finished) \
                .view(step - 1, _B_new * self.beam_size, inp_seq_len)