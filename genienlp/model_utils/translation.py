# Copyright 2021 The Board of Trustees of the Leland Stanford Junior University
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#  list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#  contributors may be used to endorse or promote products derived from
#  this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import logging
from collections import OrderedDict

import numpy as np
import torch
from transformers import SPIECE_UNDERLINE, M2M100Tokenizer

logger = logging.getLogger(__name__)


def find_overlap(start, end, used_spans):
    for i, span in enumerate(used_spans):
        span_start, span_end = span[0], span[1]
        if start <= span_end and end >= span_start:
            return i
    return -1


def count_substring(words, substring_words):
    count = 0
    beg_indices = []
    for i in range(len(words)):
        if words[i] == substring_words[0]:
            k = 0
            while k < len(substring_words) and i + k < len(words):
                if words[i + k] == substring_words[k]:
                    k += 1
                else:
                    break
            if k == len(substring_words):
                count += 1
                beg_indices.append(i)
    return count, beg_indices


def compute_attention(sample_layer_attention, att_pooling, dim=0):
    # pool attention vectors across heads
    sample_layer_attention_pooled = None
    if att_pooling == 'mean':
        sample_layer_attention_pooled = torch.mean(sample_layer_attention, dim=dim, keepdim=False)
    elif att_pooling == 'max':
        sample_layer_attention_pooled = torch.max(sample_layer_attention, dim=dim, keepdim=False)[0]

    return sample_layer_attention_pooled


def align_and_replace(src_tokens, tgt_tokens, tokenizer, sample_layer_attention_pooled, src_spans, remove_output_quotation):
    src_quotation_symbol = '"'

    # M2M100Tokenizer has missing tokens in its fixed vocabulary and encodes them as unknown (https://github.com/pytorch/fairseq/issues/3463)
    # until that's fixed we treat unknown tokens as individual words by prepending SPIECE_UNDERLINE
    if isinstance(tokenizer, M2M100Tokenizer):
        src_tokens = [token if token != tokenizer.unk_token else SPIECE_UNDERLINE + token for token in src_tokens]

    # remove padding
    if len(src_spans) % 2 != 0:
        raise ValueError(f'Corrupted span in src string: [{tokenizer.convert_tokens_to_string(src_tokens)}]')

    tgt_is_not_piece = [int(not tokenizer.is_piece_fn(token)) for token in tgt_tokens]
    tgt_piece2word_mapping = list(np.cumsum(tgt_is_not_piece) - 1)

    src_is_not_piece = [int(not tokenizer.is_piece_fn(token)) for token in src_tokens]
    src_piece2word_mapping = list(np.cumsum(src_is_not_piece) - 1)

    src_word2piece_span_mapping = OrderedDict()
    for i, j in enumerate(src_piece2word_mapping):
        if j not in src_word2piece_span_mapping:
            src_word2piece_span_mapping[j] = [i, i]
        else:
            src_word2piece_span_mapping[j][1] = i

    tgt_word2piece_span_mapping = OrderedDict()
    for i, j in enumerate(tgt_piece2word_mapping):
        if j not in tgt_word2piece_span_mapping:
            tgt_word2piece_span_mapping[j] = [i, i]
        else:
            tgt_word2piece_span_mapping[j][1] = i

    tokenizer._decode_use_source_tokenizer = True
    src_strings = tokenizer.convert_tokens_to_string(src_tokens)
    tokenizer._decode_use_source_tokenizer = False
    tgt_strings = tokenizer.convert_tokens_to_string(tgt_tokens)

    src_strings_words = src_strings.split(' ')
    tgt_strings_words = tgt_strings.split(' ')

    src_spans = [(src_spans[i], src_spans[i + 1]) for i in range(0, len(src_spans), 2)]

    src_matches = [tuple(src_strings_words[beg : end + 1]) for beg, end in src_spans]

    src_matches_counter = OrderedDict()
    for match, spans in zip(src_matches, src_spans):
        src_matches_counter.setdefault(match, []).append(spans)

    piece_tgt_spans = []

    # if translation preserved input entities we won't align them anymore
    src_spans = []
    src_matches = []
    for match, spans in src_matches_counter.items():
        if count_substring(tgt_strings_words, match)[0] != len(spans):
            for span in spans:
                src_matches.append(match)
                src_spans.append(span)
        else:
            all_indices = count_substring(tgt_strings_words, match)[1]
            for id_ in all_indices:
                beg_word, end_word = id_, id_ + len(match) - 1
                beg_piece, end_piece = tgt_word2piece_span_mapping[beg_word][0], tgt_word2piece_span_mapping[end_word][1]

                piece_tgt_spans.append((beg_piece, end_piece))

    piece_src_spans = [(src_word2piece_span_mapping[beg][0], src_word2piece_span_mapping[end][1]) for beg, end in src_spans]

    src2tgt_mapping = OrderedDict()
    for src_idx, (beg, end) in enumerate(piece_src_spans):
        topK = 1
        s1 = torch.argmax(sample_layer_attention_pooled[:, beg]).item()
        s2 = torch.argmax(sample_layer_attention_pooled[:, end]).item()

        # clamp values to max tgt_tokens length
        s1 = min(s1, len(tgt_tokens) - 1)
        s2 = min(s2, len(tgt_tokens) - 1)

        # switch tgt begin and end indices
        if s1 > s2:
            s1, s2 = s2, s1

        while find_overlap(s1, s2, piece_tgt_spans) != -1:
            overlapping_span = piece_tgt_spans[find_overlap(s1, s2, piece_tgt_spans)]
            if s1 < overlapping_span[0] and s2 < overlapping_span[1]:
                s2 = overlapping_span[0] - 1
            elif s1 >= overlapping_span[0] and s2 > overlapping_span[1]:
                s1 = overlapping_span[1] + 1
            else:
                topK += 1
                if topK >= sample_layer_attention_pooled.size(0):
                    break
                s1 = torch.topk(sample_layer_attention_pooled[:, beg], topK).indices[-1].item()
                s2 = torch.topk(sample_layer_attention_pooled[:, end], topK).indices[-1].item()

            # switch tgt begin and end indices
            if s1 > s2:
                s1, s2 = s2, s1

        piece_tgt_spans.append((s1, s2))

        src2tgt_mapping[(beg, end)] = (s1, s2)

    # update src2tgt_mapping to map to word indices in response
    for key, value in src2tgt_mapping.items():
        s1, s2 = value
        src2tgt_mapping[key] = (
            max(0, tgt_piece2word_mapping[s1]),
            min(tgt_piece2word_mapping[s2], len(tgt_tokens)),
        )

    # move through words
    tokens = []
    curr = 0
    for i, (key, value) in enumerate(src2tgt_mapping.items()):
        start, end = value
        if start > curr:
            tokens.extend(tgt_strings_words[curr:start])
        replace_match = ' '.join(src_matches[i])
        if remove_output_quotation:
            tokens.append(replace_match)
        else:
            tokens.append(src_quotation_symbol + ' ' + replace_match + ' ' + src_quotation_symbol)
        # +1 since it's inclusive
        curr = end + 1
    if curr < len(tgt_strings_words):
        tokens.extend(tgt_strings_words[curr:])

    text = ' '.join(tokens)

    return text
