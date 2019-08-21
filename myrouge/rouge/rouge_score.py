# -*- coding: utf-8 -*-
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ROUGE Metric Implementation

This is a very slightly version of:
https://github.com/pltrdy/seq2seq/blob/master/seq2seq/metrics/rouge.py

---

ROUGe metric implementation.

This is a modified and slightly extended verison of
https://github.com/miso-belica/sumy/blob/dev/sumy/evaluation/rouge.py.
"""
from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals
import re
import itertools
from collections import Counter
from nltk.stem import PorterStemmer

import os
DEBUG = os.environ.get('DEBUG', '0')

PS = PorterStemmer()

def _get_ngrams(n, text):
    """Calcualtes n-grams.

    Args:
      n: which n-grams to calculate
      text: An array of tokens

    Returns:
      A set of n-grams
    """
    ngram_counter = Counter()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_counter.update([" ".join(text[i:i + n])])
    return ngram_counter


def _split_sentence(sent):
    # sent = sent.replace('-', ' - ')
    tokens =  re.split(r'[!"#$%&\'()*+,./:;<=>?@\[\\\]\-^_`{|}~]|\s+', sent)
    tokens= [PS.stem(t) for t in tokens if t]
    return tokens


def _split_into_words(sentences):
    """Splits multiple sentences into words and flattens the result"""
    return list(itertools.chain(*[_split_sentence(_) for _ in sentences]))


def _get_word_ngrams(n, sentences):
    """Calculates word n-grams for multiple sentences.
    """
    assert len(sentences) > 0
    assert n > 0

    words = _split_into_words(sentences)
    return _get_ngrams(n, words)


def _len_lcs(x, y):
    """
    Returns the length of the Longest Common Subsequence between sequences x
    and y.
    Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence

    Args:
      x: sequence of words
      y: sequence of words

    Returns
      integer: Length of LCS between x and y
    """
    table = _lcs(x, y)
    n, m = len(x), len(y)
    return table[n, m]


def _lcs(x, y):
    """
    Computes the length of the longest common subsequence (lcs) between two
    strings. The implementation below uses a DP programming algorithm and runs
    in O(nm) time where n = len(x) and m = len(y).
    Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence

    Args:
      x: collection of words
      y: collection of words

    Returns:
      Table of dictionary of coord and len lcs
    """
    n, m = len(x), len(y)
    table = dict()
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 or j == 0:
                table[i, j] = 0
            elif x[i - 1] == y[j - 1]:
                table[i, j] = table[i - 1, j - 1] + 1
            else:
                table[i, j] = max(table[i - 1, j], table[i, j - 1])
    return table


def _recon_lcs(x, y):
    """
    Returns the Longest Subsequence between x and y.
    Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence

    Args:
      x: sequence of words
      y: sequence of words

    Returns:
      sequence: LCS of x and y
    """
    i, j = len(x), len(y)
    table = _lcs(x, y)

    def _recon(i, j):
        """private recon calculation"""
        if i == 0 or j == 0:
            return []
        elif x[i - 1] == y[j - 1]:
            return _recon(i - 1, j - 1) + [(x[i - 1], i)]
        elif table[i - 1, j] > table[i, j - 1]:
            return _recon(i - 1, j)
        else:
            return _recon(i, j - 1)

    recon_tuple = tuple(map(lambda x: x[0], _recon(i, j)))
    return recon_tuple


def _recon_lcs_with_mask(ref, dec):
    """
    return the lcs and the reference idxes of the tokens in lcs

    :param ref: sequence of words, a sentence of reference
    :param dec: sequence of words, a sentence of decoded
    :return:
        lcs: the lcs
        lcs_ref_idxes: 0-1 tuple, length=len(ref), 0 means not hit, 1 means hit by the subsequence
    """
    i, j = len(ref), len(dec)
    table = _lcs(ref, dec)

    def _recon(i, j):
        """private recon calculation"""
        if i == 0 or j == 0:
            return []
        elif ref[i - 1] == dec[j - 1]:
            # record the idx of ref
            return _recon(i - 1, j - 1) + [(ref[i - 1], i-1)]
        elif table[i - 1, j] > table[i, j - 1]:
            return _recon(i - 1, j)
        else:
            return _recon(i, j - 1)

    subseq = _recon(i, j)
    if subseq:
        lcs, lcs_ref_idxes = list(zip(*subseq))
    else:
        lcs, lcs_ref_idxes = tuple(), tuple()
    assert(len(lcs) == len(lcs_ref_idxes))
    return lcs, lcs_ref_idxes


def multi_rouge_n(sequences, scores_ids, n=2):
    """
    Efficient way to compute highly repetitive scoring
    i.e. sequences are involved multiple time

    Args:
        sequences(list[str]): list of sequences (either hyp or ref)
        scores_ids(list[tuple(int)]): list of pairs (hyp_id, ref_id)
            ie. scores[i] = rouge_n(scores_ids[i][0],
                                    scores_ids[i][1])

    Returns:
        scores: list of length `len(scores_ids)` containing rouge `n`
                scores as a dict with 'f', 'r', 'p'
    Raises:
        KeyError: if there's a value of i in scores_ids that is not in
                  [0, len(sequences)[
    """
    ngrams = [_get_word_ngrams(n, sequence) for sequence in sequences]
    counts = [sum(ngram.values()) for ngram in ngrams]

    scores = []
    for hyp_id, ref_id in scores_ids:
        evaluated_ngrams = ngrams[hyp_id]
        evaluated_count = counts[hyp_id]

        reference_ngrams = ngrams[ref_id]
        reference_count = counts[ref_id]

        overlapping_ngrams = evaluated_ngrams & reference_ngrams
        overlapping_count = sum(overlapping_ngrams.values())

        scores += [f_r_p_rouge_n(evaluated_count,
                                 reference_count, overlapping_count)]
    return scores


def rouge_n(evaluated_sentences, reference_sentences, n=2):
    """
    Computes ROUGE-N of two text collections of sentences.
    Sourece: http://research.microsoft.com/en-us/um/people/cyl/download/
    papers/rouge-working-note-v1.3.1.pdf

    Args:
      evaluated_sentences: The sentences that have been picked by the
                           summarizer
      reference_sentences: The sentences from the referene set
      n: Size of ngram.  Defaults to 2.

    Returns:
      A tuple (f1, precision, recall) for ROUGE-N

    Raises:
      ValueError: raises exception if a param has len <= 0
    """
    if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
        raise ValueError("Collections must contain at least 1 sentence.")

    evaluated_ngrams = _get_word_ngrams(n, evaluated_sentences)
    reference_ngrams = _get_word_ngrams(n, reference_sentences)
    reference_count = sum(reference_ngrams.values())
    evaluated_count = sum(evaluated_ngrams.values())

    # Gets the overlapping ngrams between evaluated and reference
    overlapping_ngrams = evaluated_ngrams & reference_ngrams
    overlapping_count = sum(overlapping_ngrams.values())

    return f_r_p_rouge_n(evaluated_count, reference_count, overlapping_count)


def f_r_p_rouge_n(evaluated_count, reference_count, overlapping_count):
    # Handle edge case. This isn't mathematically correct, but it's good enough
    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))

    return {"f": f1_score, "p": precision, "r": recall}


def _union_lcs(evaluated_sentences, reference_sentence, evaluated_left_1grams, reference_left_1grams):
    """
    Returns LCS_u(r_i, C) which is the LCS score of the union longest common
    subsequence between reference sentence ri and candidate summary C.
    For example:
    if r_i= w1 w2 w3 w4 w5, and C contains two sentences: c1 = w1 w2 w6 w7 w8
    and c2 = w1 w3 w8 w9 w5, then the longest common subsequence of r_i and c1
    is "w1 w2" and the longest common subsequence of r_i and c2 is "w1 w3 w5".
    The union longest common subsequence of r_i, c1, and c2 is "w1 w2 w3 w5"
    and LCS_u(r_i, C) = 4/5.

    Args:
      evaluated_sentences: The sentences that have been picked by the
                           summarizer
      reference_sentence: One of the sentences in the reference summaries
      evaluated_left_1grams: Left 1grams of evaluated sentences (all the sentences)
      reference_left_1grams: Left 1grams of reference sentences (all the sentences)

    Returns:
      float: LCS_u(r_i, C)

    ValueError:
      Raises exception if a param has len <= 0
    """
    if len(evaluated_sentences) <= 0:
        raise ValueError("Collections must contain at least 1 sentence.")

    reference_words = _split_into_words([reference_sentence])
    hit = 0

    for eval_s in evaluated_sentences:
        evaluated_words = _split_into_words([eval_s])
        lcs, lcs_ref_idxes = _recon_lcs_with_mask(reference_words, evaluated_words)
        if DEBUG != '0':
            print(lcs)
        for idx in lcs_ref_idxes:
            token = reference_words[idx]
            if evaluated_left_1grams[token] > 0 and reference_left_1grams[token] > 0:
                hit += 1
                evaluated_left_1grams[token] -= 1
                reference_left_1grams[token] -= 1

    return hit


def rouge_l_summary_level(evaluated_sentences, reference_sentences, alpha=0.5):
    """
    Computes ROUGE-L (summary level) of two text collections of sentences.
    http://research.microsoft.com/en-us/um/people/cyl/download/papers/
    rouge-working-note-v1.3.1.pdf

    Calculated according to:
    R_lcs = SUM(1, u)[LCS<union>(r_i,C)]/m
    P_lcs = SUM(1, u)[LCS<union>(r_i,C)]/n
    F_lcs = ((1 + beta^2)*R_lcs*P_lcs) / (R_lcs + (beta^2) * P_lcs)

    where:
    SUM(i,u) = SUM from i through u
    u = number of sentences in reference summary
    C = Candidate summary made up of v sentences
    m = number of words in reference summary
    n = number of words in candidate summary

    Args:
      evaluated_sentences: The sentences that have been picked by the
                           summarizer
      reference_sentence: One of the sentences in the reference summaries

    Returns:
      A float: F_lcs

    Raises:
      ValueError: raises exception if a param has len <= 0
    """
    if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
        raise ValueError("Collections must contain at least 1 sentence.")

    # total number of words in reference sentences
    reference_tokens = _split_into_words(reference_sentences)
    m = len(reference_tokens)
    reference_left_1grams = _get_ngrams(1, reference_tokens)

    # total number of words in evaluated sentences
    evaluated_tokens = _split_into_words(evaluated_sentences)
    n = len(evaluated_tokens)
    evaluated_left_1grams = _get_ngrams(1, evaluated_tokens)

    # print("m,n %d %d" % (m, n))
    union_hit_sum = 0
    for ref_s in reference_sentences:
        # the 1grams will be updated in the function
        union_hit = _union_lcs(evaluated_sentences, ref_s, evaluated_left_1grams,
                                     reference_left_1grams)
        union_hit_sum += union_hit

    if DEBUG != '0':
        print(m, n, union_hit_sum)
    llcs = union_hit_sum
    # avoid division by zero
    r_lcs = llcs / (m + 1e-12)
    p_lcs = llcs / (n + 1e-12)
    if alpha == None:
        beta = p_lcs / (r_lcs + 1e-12)
        num = (1 + (beta**2)) * r_lcs * p_lcs
        denom = r_lcs + ((beta**2) * p_lcs)
        f_lcs = num / (denom + 1e-12)
    else:
        num = p_lcs * r_lcs
        denom = (1 - alpha) * p_lcs + alpha * r_lcs
        if denom == 0:
            f_lcs = 0.0
        else:
            f_lcs = num / denom
    return {"f": f_lcs, "p": p_lcs, "r": r_lcs}
