import codecs
import itertools
import numpy as np
import time

from collections import defaultdict
from sklearn.utils import shuffle
from sklearn.metrics import classification_report

BOS = u'<s>'
EOS = u'</s>'

np.random.seed(6778)


def read_ruscorpora(file_path):
    corpora = [[]]
    with codecs.open(file_path, encoding='utf-8') as inf:
        for line in inf:
            line = line.strip('\r\n ')
            if '\t' in line:
                line = line[:line.find('\t')]
            if line.startswith('#'):
                continue
            if not line:
                corpora.append([])
                continue
            wordform, lemma, gram = line.rsplit('/', 2)
            pos = gram[:gram.find('=')] if '=' in gram else gram
            pos = pos.split(',')[0]
            corpora[-1].append((wordform.lower(), pos, lemma, gram))

    return corpora


def get_trans_prob(prev_next_tag, prob):
    return np.log(prob[prev_next_tag]) if prev_next_tag in prob else -np.inf


def get_word_prob(word_tag, prob):
    return np.log(prob[word_tag]) if word_tag in prob else -np.inf


def viterbi_decode(corpus_sent, tags, t_prob, w_prob):
    sent = [item[0] for item in corpus_sent]
    viterbi = np.zeros((len(sent), len(tags)))
    back_point = np.zeros((len(sent), len(tags)), dtype=np.int64)

    # initial values
    viterbi[0, :] = [get_trans_prob((BOS, t), t_prob) + get_word_prob((sent[0], t), w_prob) for t in tags]

    for i, word in enumerate(sent):  # loop through each word
        if not i:
            continue
        for j, t in enumerate(tags):  # loop through each possible tag for this word
            prob = np.full((len(tags),), get_word_prob((word, t), w_prob))
            for k, prev_tag in enumerate(tags):  # look behind on each possible transition
                prob[k] += viterbi[i - 1, k] + get_trans_prob((prev_tag, t), t_prob)
            viterbi[i, j] = np.amax(prob)
            back_point[i, j] = np.argmax(prob)

    # find sequence with maximum probability
    sequence = [0] * len(sent)
    sequence[-1] = np.argmax(viterbi[-1, :])
    for i in reversed(range(1, len(sent))):
        sequence[i - 1] = back_point[i, sequence[i]]

    # return list(zip(sent, [tags[int(i)] for i in sequence]))
    return [tags[int(i)] for i in sequence]


def get_tags_and_probs(corpora):
    cnt_pos_pos = defaultdict(int)
    cnt_word_pos = defaultdict(int)
    cnt_pos = defaultdict(int)

    for sent in corpora:
        if not len(sent):
            continue
        cnt_pos[BOS] += 1
        items = [[BOS, BOS]] + sent + [[EOS, EOS]]
        for ind, word in enumerate(items[1:]):
            word_pos = (word[0], word[1])
            pos_pos = (items[ind][1], word[1])

            if pos_pos == (BOS, EOS):
                print(items)

            cnt_pos[word[1]] += 1
            cnt_pos_pos[pos_pos] += 1
            cnt_word_pos[word_pos] += 1

    # transition probabilities
    t_prob = {key: float(cnt_pos_pos[key]) / cnt_pos[key[0]] for key in cnt_pos_pos}
    # emission probabilities
    w_prob = {key: float(cnt_word_pos[key]) / cnt_pos[key[1]] for key in cnt_word_pos}

    # decode random sentence
    tags = list(cnt_pos.keys())
    tags.remove(BOS)
    tags.remove(EOS)

    return tags, w_prob, t_prob


if __name__ == '__main__':
    offset = 80000
    start = time.time()
    corpus = read_ruscorpora('ruscorpora.parsed.txt')
    print('Total sentences: %s' % len(corpus))

    corpus = shuffle(corpus)
    corpus_train, corpus_test = corpus[:offset], corpus[offset:]

    tags, word_prob, trans_prob = get_tags_and_probs(corpus_train)
    print("Corpora loaded: %ss" % (time.time() - start))

    start = time.time()
    label_true = []
    label_pred = []
    for _i, s in enumerate(corpus_test):
        if not _i % 1000:
            print(_i)

        label_true.append([item[1] for item in s])
        label_pred.append(viterbi_decode(s, tags, trans_prob, word_prob))

    # print result
    with open('viterbi_res.txt', mode='w') as res_f:
        for _i, s in enumerate(corpus_test):
            s_true = '\t'.join(['%s-%s' % (item[0], item[1]) for item in s])
            s_pred = '\t'.join(['%s-%s' % (item[0], label_pred[_i][j]) for j, item in enumerate(s)])
            res_f.write('%s\n%s\n\n' % (s_true, s_pred))

    print("Corpora decoded: %ss" % (time.time() - start))
    print(classification_report(list(itertools.chain(*label_true)), list(itertools.chain(*label_pred))))
