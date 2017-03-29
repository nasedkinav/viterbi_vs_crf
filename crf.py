import itertools
import numpy as np
import time

from sklearn.utils import shuffle
from sklearn_crfsuite import CRF
from sklearn.metrics import classification_report

np.random.seed(6778)


def transform_word(prev, prev_t, curr, _next):
    features = {
        'word.lower()': curr.lower(),
        'word[3:]': curr[3:],
        'word[2:]': curr[2:],
        'word[-3:]': curr[-3:],
        'word[-2:]': curr[-2:],
        'word.isupper()': curr.isupper(),
        'word.istitle()': curr.istitle(),
        'word.isdigit()': curr.isdigit()
    }

    if prev and prev_t:
        features.update({
            '-1:word.lower()': prev.lower(),
            '-1:word.isupper()': prev.isupper(),
            '-1:word.istitle()': prev.istitle(),
            '-1:word.isdigit()': prev.isdigit(),
            '-1:postag': prev_t,
        })
    else:
        features['BOS'] = True

    if _next:
        features.update({
            '+1:word.lower()': _next.lower(),
            '+1:word.isupper()': _next.isupper(),
            '+1:word.istitle()': _next.istitle(),
            '+1:word.isdigit()': _next.isdigit(),
        })
    else:
        features['EOS'] = True

    return features


def transform_corpus(data):
    X, y = [], []
    for sent in data:
        for i, item in enumerate(sent):
            X.append([transform_word(
                None if not i else sent[i - 1][0],
                None if not i else sent[i - 1][1],
                item[0],
                None if i == len(sent) - 1 else sent[i + 1][0]
            )])
            y.append([item[1]])

    return X, y


def read_ruscorpora(file_path):
    sent = [[]]
    with open(file_path, encoding='utf-8') as inf:
        for line in inf:
            line = line.strip('\r\n ')
            if '\t' in line:
                line = line[:line.find('\t')]
            if line.startswith('#'):
                continue
            if not line:
                sent.append([])
                continue
            wordform, lemma, gram = line.rsplit('/', 2)
            pos = gram[:gram.find('=')] if '=' in gram else gram
            pos = pos.split(',')[0]
            sent[-1].append((wordform, pos, lemma, gram))

    return sent


if __name__ == '__main__':
    offset = 80000
    start = time.time()
    corpus = read_ruscorpora('ruscorpora.parsed.txt')
    corpus = shuffle(corpus)
    corpus_train, corpus_test = corpus[:offset], corpus[offset:]
    print("Corpora loaded: %ss" % (time.time() - start))

    start = time.time()
    X_train, y_train = transform_corpus(corpus_train)
    print("Corpora transformed: %ss" % (time.time() - start))

    crf = CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    crf.fit(X_train, y_train)

    # form predictions
    y_test = []
    y_pred = []
    for s in corpus_test:
        y_s_pred = []
        y_s_test = []
        for i, item in enumerate(s):
            y_s_test.append(item[1])
            x = transform_word(
                None if not i else s[i - 1][0],
                None if not i else y_s_pred[i - 1],
                item[0],
                None if i == len(s) - 1 else s[i + 1][0]
            )
            y_s_pred.append(crf.predict([[x]])[0][0])
        y_test.append(y_s_test)
        y_pred.append(y_s_pred)

    # print result
    with open('crf_res.txt', mode='w') as res_f:
        for _i, s in enumerate(corpus_test):
            s_true = '\t'.join(['%s-%s' % (item[0], item[1]) for item in s])
            s_pred = '\t'.join(['%s-%s' % (item[0], y_pred[_i][j]) for j, item in enumerate(s)])
            res_f.write('%s\n%s\n\n' % (s_true, s_pred))

    print(classification_report(list(itertools.chain(*y_test)), list(itertools.chain(*y_pred))))
