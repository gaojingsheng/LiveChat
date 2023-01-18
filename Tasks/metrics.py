from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize 
from rouge import Rouge

from collections import Counter
import json

# generate_corpus: [sentence1, sentence2, ...]
# reference_corpus: [sentence1, sentence2, ...]
# use eval_result to retrive the bleu, dist, rouge result

def eval_result(generate_corpus, reference_corpus):
    
    generate_corpus = [" ".join([s for s in gen]) for gen in generate_corpus]
    reference_corpus = [" ".join([s for s in ref]) for ref in reference_corpus]

    results = {}
    bleu_result = cal_bleu(generate_corpus, reference_corpus)
    dist_result = cal_dist(generate_corpus, reference_corpus)
    rouge_result = cal_rouge(generate_corpus, reference_corpus)

    results.update(bleu_result)
    results.update(dist_result)
    results.update(rouge_result)

    return results

def cal_bleu(generate_corpus, reference_corpus):

    ngrams = ['bleu-{}'.format(n) for n in range(1, 5)]
    ngram_weights = []
    results = {}
    for ngram in ngrams:
        results[ngram] = []
    for n in range(1, 5):
        weights = [0.] * 4
        weights[:n] = [1. / n] * n
        ngram_weights.append(weights)

    for gen, refs in zip(generate_corpus, reference_corpus):
        gen = word_tokenize(gen.strip())
        refs = word_tokenize(refs.strip())
        try:
            for ngram, weights in zip(ngrams, ngram_weights):
                score = sentence_bleu([refs], gen, weights=weights, smoothing_function=SmoothingFunction().method7)
                assert type(score) == float or int
                results[ngram].append(score * 100)
        except:
            pass    
    for item in results:
        results[item] = sum(results[item])/len(results[item])
    return results

def cal_dist(generate_corpus, reference_corpus=None):

    results = {}
    ngrams_all = [Counter() for _ in range(4)]
    for gen in generate_corpus:
        gen = gen.strip()
        ngrams = []
        for i in range(4):
            ngrams.append(gen[i:])
            ngram = Counter(zip(*ngrams))
            ngrams_all[i].update(ngram)
    for i in range(4):
        results[f'distinct-{i+1}'] = (len(ngrams_all[i])+1e-12) / (sum(ngrams_all[i].values())+1e-5) * 100 
    return results

def cal_rouge(generate_corpus, reference_corpus):

    rouge = Rouge()
    results = rouge.get_scores(generate_corpus, reference_corpus, avg=True)
    for key in results:
        results[key]=results[key]['f']*100
    return results

if __name__ == "__main__":
    evaluation_path = ""
    total_data = json.load(open(evaluation_path,"rb"))
    generate_corpus = []
    reference_corpus = []
    print("total_data length is ", len(total_data))
    for item in total_data:
        generate_corpus.append(item["generation"])
        reference_corpus.append(item["response"])
        
    generate_corpus = [" ".join([s for s in gen[:62]]) for gen in generate_corpus]
    reference_corpus = [" ".join([s for s in ref[:62]]) for ref in reference_corpus]
    
    bleu_result = cal_bleu(generate_corpus, reference_corpus)
    dist_result = cal_dist(generate_corpus, reference_corpus)
    rouge_result = cal_rouge(generate_corpus, reference_corpus)

    print(bleu_result)
    print(dist_result)
    print(rouge_result)







