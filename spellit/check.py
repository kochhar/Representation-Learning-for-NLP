from __future__ import print_function
import string
import sys

import Levenshtein as lev

class NGramSpellChecker(object):
    """A simple NGram based spell checker."""

    def __init__(self, prefix_model, max_length=3):
        """
        Params:
        prefix_model: an ngram prefix model
        max_length: the maximum length of the ngrams indexed in the prefix
                    model
        """
        self.max_length = max_length
        self.prefix_model = prefix_model

    def suggest(self, text, max=5):
        """Checks text for spelling suggestions at the end."""
        text = text.translate(None, string.punctuation)
        tokens = text.split()
        if not tokens: return []

        token, pre_window = tokens[-1], tokens[:-1]

        # First check if the current token plus a window of priors exist in the
        # vocabulary. If they do, then assume there's no spelling mistake
        if self.known(token, pre_window):
            return []

        # If unknown, assume it's a spelling error and find the best completion
        # given the pre_text.
        candidates = self.candidates(pre_window)
        closest = self.closest(token, candidates)
        return [cand for (_, cand) in closest[:max]]

    def known(self, token, pre_window):
        model = self.prefix_model
        for snippet in self.snippets(token, pre_window):
            count, _ = model.completions(snippet)
            if count: return True

        return False

    def candidates(self, tokens):
        """Given a sequnce of tokens, generates candidates for the next word."""
        model = self.prefix_model
        if not tokens:
            _, completions = model.completions([])
            return completions.keys()

        candidates = []
        for snippet in self.snippets(tokens[-1], tokens[:-1]):
            _, completions = model.completions(snippet)
            candidates.extend(completions.keys())
        return candidates

    def closest(self, token, candidates):
        sim_metric = lev.jaro_winkler
        similarities = sorted([(sim_metric(token, cand), cand) for cand in candidates],
                              reverse=True)
        return similarities

    def snippets(self, token, pre_window):
        """Given a token and a pre_window yields snippets of decreasing length,
        stopping with just the token."""
        max_window = len(pre_window)

        # If the token sequence has a count > 0, assume it's a known word
        for winlen in range(-max_window, 0):
            yield pre_window[winlen:] + [token]
        yield [token]



class NGramPrefixModel(object):
    """Defines an n-gram model to search a for completions given a prefix."""

    def __init__(self, ngrams):
        """Initalise the model with an iterable of ngrams and their counts.

        Parms:
          - ngrams: an iterable containing tuples of the form
                    ((tok1, tok2, ... , tokN), count)"""
        self.ngrams = _ngram_prefix_trie(ngrams)

    def completions(self, prefix_tokens):
        """Given a string prefix, finds the most likely completions of the
        prefix along with their counts."""
        completion_tree = self.ngrams
        if not prefix_tokens:
            return None, completion_tree

        for token in prefix_tokens:
            count, completion_tree = completion_tree.get(token, (0, {}))

        return (count, completion_tree)


def _ngram_prefix_trie(ngrams):
    """Converts an iterable of ngram tuples and their counts to a ngram trie.

    Params:
       - ngrams: an iterable containing tuples of the form
                 ((tok1, tok2, ... , tokN), count)
    """
    trie = {}

    for ngram, count in ngrams:
        next_trie = trie
        for tok in ngram:
            if tok not in next_trie:
                next_trie[tok] = (0, {})

            curr_count, continuations = next_trie[tok]
            next_trie[tok] = (curr_count+count, continuations)
            next_trie = continuations

    return trie


def _read_ngrams(filepath, max_ngrams=1000, min_freq=10):
    """Reads ngrams and frequencies from filepath and returns a generator which
    will yield the ngrams. The ngrams read are filtered based on some rules.

    1. ngrams which contain #EOS# are omitted
    2. ngrams which contain a punctuation only token are omitted
    3. ngrams which have frequency less than min_freq are omitted

    Params:
        filepath: the file containing the ngrams
        max_ngrams: the maximum number of ngrams to read, will stop once this has been reached
                    min_freq
        min_freq: the mininum frequency for a ngram to be preserved.
    """

    read = 0
    with open(filepath, 'r') as filehand:
        for line in filehand:
            parts = line.split('\t')

            # Skip any lines that don't have at least 2 parts
            if len(parts) < 2: continue

            count, tokens = int(parts[0]), parts[1:]
            # Skip any n-grams which don't occur at-least 5 times
            if count < min_freq: continue
            # Strip #EOS#
            tokens = [t if not t == '#EOS#' else '' for t in tokens]
            # Strip punctuation
            tokens = [t.translate(None, string.punctuation) for t in tokens]
            # Strip newlines
            tokens = [t.strip('\n') for t in tokens]
            # Skip any n-gram which has an empty token after cleaning
            if not all(tokens): continue

            yield (tokens, count)
            read += 1
            if max_ngrams and read > max_ngrams:
                break
            if read % 250000 == 0:
                print ('.', end='', file=sys.stderr)
