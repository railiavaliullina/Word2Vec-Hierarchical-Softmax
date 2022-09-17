import numpy as np
from numpy import vstack, array, dot, argsort
from gensim import matutils

REAL = np.float32


class Evaluation:
    def __init__(self, model, vocab, index_to_word, word2id):
        self.syn0 = model.w1
        self.vocab = vocab
        self.index2word = index_to_word
        self.word2id = word2id

    def init_sims(self):
        if getattr(self, 'syn0norm', None) is None:
            print("precomputing L2-norms of word weight vectors")
            self.syn0norm = vstack(matutils.unitvec(vec) for vec in self.syn0).astype(REAL)

    def most_similar(self, positive, negative, topn=False):
        """
            Find the top-N most similar words. Positive words contribute positively towards the
            similarity, negative words negatively.

            This method computes cosine similarity between a simple mean of the projection
            weight vectors of the given words, and corresponds to the `word-analogy` and
            `distance` scripts in the original word2vec implementation.

            Example::

              # >>> trained_model.most_similar(positive=['woman', 'king'], negative=['man'])
              [('queen', 0.50882536), ...]

            """
        self.init_sims()

        # add weights for each word, if not already present; default to 1.0 for positive and -1.0 for negative words
        positive = [(word, 1.0) if isinstance(word, str) else word for word in positive]
        negative = [(word, -1.0) if isinstance(word, str) else word for word in negative]
        all_words, mean = set(), []
        for word, weight in positive + negative:
            if word in self.vocab:
                mean.append(weight * matutils.unitvec(self.syn0[self.word2id[word]]))
                all_words.add(self.word2id[word])
            else:
                print("word '%s' not in vocabulary; ignoring it" % word)
        if not mean:
            raise ValueError("cannot compute similarity with no input")
        mean = matutils.unitvec(array(mean).mean(axis=0)).astype(REAL)
        dists = dot(self.syn0norm, mean)
        if not topn:
            return dists
        best = argsort(dists)[::-1][:topn + len(all_words)]
        # ignore (don't return) words from the input
        result = [(self.index2word[sim], dists[sim]) for sim in best if sim not in all_words]
        return result[:topn]

    def accuracy(self, questions):
        """
        Compute accuracy of the model. `questions` is a filename where lines are
        4-tuples of words, split into sections by ": SECTION NAME" lines.
        See https://code.google.com/p/word2vec/source/browse/trunk/questions-words.txt for an example.

        The accuracy is reported (=printed to log and returned as list) for each
        section separately, plus there's one aggregate summary at the end.

        Use `restrict_vocab` to ignore all questions containing a word whose frequency
        is not in the top-N most frequent words (default top 30000).

        This method corresponds to the `compute-accuracy` script of the original C word2vec.

        """

        def log_accuracy(section):
            correct, incorrect = section['correct'], section['incorrect']
            if correct + incorrect > 0:
                print("%s: %.1f%% (%i/%i)" %
                      (section['section'], 100.0 * correct / (correct + incorrect),
                       correct, correct + incorrect))

        sections, section = [], None
        file_ = open(questions)
        for line_no, line in enumerate(file_):
            if line_no % 1000 == 0:
                print(f'test: {line_no}')
            if line.startswith(': '):
                # a new section starts => store the old section
                if section:
                    sections.append(section)
                    log_accuracy(section)
                section = {'section': line.lstrip(': ').strip(), 'correct': 0, 'incorrect': 0}
            else:
                if not section:
                    raise ValueError("missing section header before line #%i in %s" % (line_no, questions))

                a, b, c, expected = [word.lower() for word in line.split()]

                if self.word2id.get(a, None) is None or self.word2id.get(b, None) is None \
                        or self.word2id.get(c, None) is None or self.word2id.get(expected, None) is None:
                    continue
                predicted, ignore = None, set(self.word2id[v] for v in [a, b, c])
                w = self.most_similar(positive=[b, c], negative=[a], topn=False)
                for index in argsort(w)[::-1]:
                    if index not in ignore:
                        predicted = self.index2word[index]
                        break
                section['correct' if predicted == expected else 'incorrect'] += 1

        if section:
            sections.append(section)
            log_accuracy(section)

        total = {'section': 'total', 'correct': sum(s['correct'] for s in sections),
                 'incorrect': sum(s['incorrect'] for s in sections)}
        log_accuracy(total)
        sections.append(total)
        print(f'Accuracy: {total}, total: {total["correct"] / (total["correct"] + total["incorrect"])}')
        return sections
