from itertools import combinations
from collections import deque

# Flask
from flask import Flask, request, jsonify
from flask_cors import CORS

# NLP: http://www.nltk.org/howto/stem.html
from utils import model
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer('english')

app = Flask(__name__)
CORS(app)

# Similar words to check for
LIMIT = 30

# instructions: export FLASK_APP=server.py
# flask run

hints_used = deque()


def bestMatchPair(words):
    return set(pair[0] for pair in model.most_similar(positive=words, topn=LIMIT))


def bestMatch(words_own, words_other, words_gray, words_black, allWords):
    global hints_used

    print(f'\n finding hints for: {words_own}')
    print(f'\n excluding previous hints: {hints_used}')
    similar_words = set()

    for size in range(1, 4):
        for combo in combinations(words_own, size):
            # print(combo, similar_words, '\n')
            similar_words |= bestMatchPair(combo)

    # only take non-compound words that don't share a stem
    # todo: OUTSOURCE, add cors restrictions
    similar_words = (w for w in similar_words
                     if w.isalpha()
                     and stemmer.stem(w) not in (stemmer.stem(x) for x in allWords)
                     and all(w not in x and x not in w for x in allWords)
                     and w.upper() != w
                     and w not in hints_used)
    #print(similar_words, 'after')

    def getWeights(w):
        if w in words_own:
            return 1
        elif w in words_gray:
            return -1
        elif w in words_other:
            return -2
        elif w in words_black:
            return -10
        else:
            return 0

    # todo: better function names, comments, consistency
    def get_danger(hint):
        similarity = sorted([(model.wv.similarity(hint, w), w) for w in allWords], reverse=True)
        # type: float: sim, str: word

        score = sum(match[0] ** 2 * getWeights(match[1]) for match in similarity)
        #scores.append((score, hint, similarity[:2]))
        return score

    scores = []

    for hint in similar_words:
        similarity = sorted([(model.wv.similarity(hint, w), w) for w in allWords], reverse=True)
        #print(hint, '#', similarity)
        total_sim = 0.0
        for i, match in enumerate(similarity):
            #print(i, match) todo make constant
            if not(match[0] > .25 and match[1] in words_own):
                scores.append((i * (1 + get_danger(hint)/2) + total_sim / (i + .0001), hint, similarity[:i]))
                break
            else:
                total_sim += match[0]

    scores.sort(reverse=True)

    print(*scores[:20], sep='\n')

    # Scores: (float: score, str: word, list[str]: targets)
    best = scores[0][1]
    targets = [s[1] for s in scores[0][2]]
    #print(targets)

    hints_used.append(best)
    if len(hints_used) > 20:
        hints_used.popleft()

    return best, targets, {t: model.wv.similarity(best, t) for t in targets}, \
           [model.wv.similarity(best, w) for w in allWords]


@app.route('/')
def hello_world():
    return 'Hey'


@app.route('/api/', methods=['POST'])
def parseWords():
    words = request.get_json()

    hint, targets, dist, allDists = bestMatch(words['red'], words['blue'], words['gray'], words['black'], words['allWords'])

    return jsonify({'hint': hint, 'targets': targets, 'dist': dist, 'allDists': allDists})
