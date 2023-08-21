from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize # word_tokenize according to blank
from rouge import Rouge

from collections import Counter
import json
import pickle as pk
import numpy as np
# import torch

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

    print("generate_corpus is:",generate_corpus[:10])
    print("reference_corpus is:",reference_corpus[:10])
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
        # print(gen, refs)
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

def cal_ppl(input_list):
    # this is a fault cal_ppl
    return 1

if __name__ == "__main__":
    eva_generation_path = "generation.json"
    total_data = json.load(open(eva_generation_path,"rb"))
    # dev_data = pk.load(open("dataset_withpersona/dev_data.pk","rb"))
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



# {'bleu-1': 31.59630122915178, 'bleu-2': 20.18833827479204, 'bleu-3': 12.927942535617492, 'bleu-4': 8.251670544064492}
# {'distinct-1': 0.9541625837141989, 'distinct-2': 1.978280561728691, 'distinct-3': 10.446433473700157, 'distinct-4': 18.939314718827234}
# {'rouge-1': 25.18328101682556, 'rouge-2': 6.363913083144415, 'rouge-l': 23.290554644714607}



