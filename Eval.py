## 使用BLEU对生成的句子进行评估
import pandas as pd
import os
import numpy as np
import math
import collections
from tqdm import tqdm

def bleu(pred_seq, label_seq, k):
    """Compute the BLEU.
    """
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score

if __name__ == "__main__":

    id = "top_p"

    label_path = "./data/ROCStories_test.csv"
    pred_path = f"./output/{id}.csv"
    
    df_label = pd.read_csv(label_path)
    df_pred = pd.read_csv(pred_path)

    label = df_label.values
    pred = df_pred.values
    
    print("Loading label data...")
    label_sents = []
    for line in label:
        sentences = line[3:]
        sentence = ""
        for sent in sentences:
            sentence += " "
            sentence += sent
        # print(sentence)
        # input("Press Enter...")
        label_sents.append(sentence)
    print(f"Done. {len(label_sents)} sentences loaded.")

    print("Loading pred data...")
    pred_sents = []
    for line in pred:
        sentences = line
        sentence = ""
        for sent in sentences:
            sentence += " "
            sentence += str(sent)
        # print(sentence)
        # input("Press Enter...")
        pred_sents.append(sentence)
    print(f"Done. {len(pred_sents)} sentences loaded.")

    scores = []
    length = len(label_sents)

    tqdm_iterator = tqdm(range(length), dynamic_ncols=True, desc=f'Evaluating')
    for i in tqdm_iterator:
        score = bleu(pred_sents[i], label_sents[i], 1)
        scores.append(score)
        tqdm_iterator.set_postfix({'CurScore': score})
    print(f"BLEU-1: {np.mean(scores)}")

    with open(f"eval_{id}_output.csv", "w", encoding="utf-8") as cf:
        ef = pd.DataFrame(columns=["label", "pred", "bleu"])
        for i in range(length):
            ef.loc[len(ef.index)] = [label_sents[i], pred_sents[i], scores[i]]
        ef.to_csv(cf, index=False, header=True, encoding="utf-8")
    
    print(f"Finish evaluating {len(ef.index)} groups of stories. Results saved in eval_{id}_output.csv.")