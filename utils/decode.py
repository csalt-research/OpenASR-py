import torch
import torch.nn as nn
import torch.nn.functional as F
from queue import PriorityQueue

class BeamSearchNode(object):
    def __init__(self, hidden_state, previous_node, \
                 token_id, log_prob, length):
        self.h = hidden_state
        self.prev_node = previous_node
        self.token_id = token_id
        self.logp = log_prob
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        # TODO: Add here a function for shaping a reward
        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward

    def __lt__(self, _):
        # Called when `score` of two nodes is same when
        # they are being inserted into a PriorityQueue
        return True

def beam_decode(model, vocab, src, lengths, opts, device):
    with torch.no_grad():
        # F-prop through encoder to get state
        encoder_final, memory_bank, new_lengths = model.encoder(src, lengths)
        encoder_final = model.bridge(encoder_final)
        memory_bank = model.W(memory_bank)

        # Parameters
        beam_width = opts.beam_width
        topk = opts.nbest
        SOS_token = vocab.encode('<sos>')
        EOS_token = vocab.encode('<eos>')

        decoded_batch = []

        # Decode sentence by sentence
        for idx in range(src.size(1)):
            enc_f = tuple([h[:, idx, ...].unsqueeze(1) for h in encoder_final])
            mem_b = memory_bank[:, idx, ...].unsqueeze(1)
            nl = new_lengths[idx].unsqueeze(0)
            model.decoder.init_state(enc_f)

            # Start with the start of the sentence token
            decoder_input = torch.LongTensor([[SOS_token]]).to(device)

            # Number of sentence to generate
            endnodes = []
            number_required = min((topk + 1), topk - len(endnodes))

            # Initialize start node
            node = BeamSearchNode(model.decoder.state, None, decoder_input, 0, 1)
            nodes = PriorityQueue()

            # Start the queue
            nodes.put((-node.eval(), node))
            qsize = 1

            # Start beam search
            while True:
                if qsize > 2000: 
                    break

                # Fetch the best node
                score, n = nodes.get()
                decoder_input = n.token_id
                decoder_state = n.h

                if n.token_id.item() == EOS_token and n.prev_node != None:
                    endnodes.append((score, n))
                    if len(endnodes) >= number_required:
                        break
                    else:
                        continue

                # Decode for one step using decoder
                model.decoder.state = decoder_state
                decoder_output, _ = model.decoder(decoder_input, mem_b, nl)

                log_prob, indexes = torch.topk(decoder_output.squeeze(), beam_width)
                nextnodes = []

                for new_k in range(beam_width):
                    decoded_t = indexes[new_k].view(1, -1)
                    log_p = log_prob[new_k].item()

                    node = BeamSearchNode(model.decoder.state, n, decoded_t, n.logp + log_p, n.leng + 1)
                    score = -node.eval()
                    nextnodes.append((score, node))

                # Put them into queue
                for score, n in nextnodes:
                    nodes.put((score, n))
                qsize += len(nextnodes) - 1

            # Choose nbest paths, back trace them
            if len(endnodes) == 0:
                endnodes = [nodes.get() for _ in range(topk)]

            utterances = []
            for score, n in sorted(endnodes, key=lambda x: x[0]):
                utterance = []
                log_p = n.logp
                utterance.append(n.token_id.item())
                while n.prev_node != None:
                    n = n.prev_node
                    utterance.append(n.token_id.item())
                utterance = utterance[::-1]
                utterances.append((utterance, log_p))

            decoded_batch.append(utterances)

        return decoded_batch


# def greedy_decode(model, vocab, src, lengths, opts, device):
#     batch_size, seq_len = target_tensor.size()
#     decoded_batch = torch.zeros((batch_size, MAX_LENGTH))
#     decoder_input = torch.LongTensor([[SOS_token] for _ in range(batch_size)], device=device)

#     for t in range(MAX_LENGTH):
#         decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)

#         topv, topi = decoder_output.data.topk(1)  # get candidates
#         topi = topi.view(-1)
#         decoded_batch[:, t] = topi

#         decoder_input = topi.detach().view(-1, 1)

#     return decoded_batch
