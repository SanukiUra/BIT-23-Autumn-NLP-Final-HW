import torch
import torch.nn as nn
import time
import os
import pandas as pd
from functorch.einops import rearrange
from tqdm import tqdm
from torch.utils.data import DataLoader

from MyDataset import Corpus
from Model import Transformer


def gen_mask(length):
    pass
    mask = rearrange(torch.triu(torch.ones(length, length)) == 1, 'h w -> w h')
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def decode(src, src_key_padding_mask, max_len, decoder = "top_p", p = 0.9, k = 5, t = 1.0):
    flag = False
    src_mask = gen_mask(src.shape[1]).to(device)
    memory_key_padding_mask = src_key_padding_mask.clone()
    sos = 1
    eos = 2

    sentences = []
    
    for num_sents in range(4):

        new_sents = torch.ones(1, 1).type_as(src.data).fill_(sos)
        if num_sents == 0:
            out = torch.ones(1, 1).type_as(src.data).fill_(sos)
        else:
            out = torch.cat((out, torch.ones(1, 1).type_as(src.data).fill_(sos)), 1)

        for i in range(max_len - 1):
            tgt_mask = gen_mask(out.shape[1]).to(device)
            tgt = model(src, out, src_mask, tgt_mask, src_key_padding_mask, None, memory_key_padding_mask)
            if decoder == "greedy":
                next_word = tgt[:, -1, :]
                next_word = torch.argmax(next_word, dim=-1).squeeze(-1)
                
            elif decoder == "top_p":
                # 做TOP-P
                next_word = top_p(tgt[:, -1, :], p)
                # 做采样
                probabilities = torch.softmax(temperature(next_word, t), dim=-1)
                next_word = torch.multinomial(probabilities, num_samples=1).squeeze(-1).squeeze(-1)

            elif decoder == "top_k":
                # 做TOP-K
                next_word = top_k(tgt[:, -1, :], k)
                # 做采样
                probabilities = torch.softmax(temperature(next_word, t), dim=-1)
                next_word = torch.multinomial(probabilities, num_samples=1).squeeze(-1).squeeze(-1)

            elif decoder == "top_k_p":
                next_word = top_k(top_p(tgt[:, -1, :], p),k )

                probabilities = torch.softmax(temperature(next_word, t), dim=-1)
                next_word = torch.multinomial(probabilities, num_samples=1).squeeze(-1).squeeze(-1)
            
            elif decoder == "temp_only":
                # print("decoder chosen as 'temp_only'")
                next_word = tgt[:, -1, :]

                probabilities = torch.softmax(temperature(next_word, t), dim=-1)
                next_word = torch.multinomial(probabilities, num_samples=1).squeeze(-1).squeeze(-1)

            new_sents = torch.cat((new_sents, torch.ones(1, 1).type_as(src.data).fill_(next_word)), 1)
            if next_word == 2:
                out = torch.cat((out, torch.ones(1, 1).type_as(src.data).fill_(2)), 1)
                break
            if out.shape[-1] >= 128:
                break
            out = torch.cat((out, torch.ones(1, 1).type_as(src.data).fill_(next_word)), 1)

        sentences.append(new_sents)

        if out.shape[-1] >= 128:
            flag = True
            break

    return sentences, flag

# BeamSearch是有问题的
def beam_search(src, src_key_padding_mask, max_len):
    src_mask = gen_mask(src.shape[1]).to(device)
    memory_key_padding_mask = src_key_padding_mask.clone()

    k = 1
    sos = 1
    eos = 2
    vocab_size = dataset.vocab_size

    k_prev_words = torch.full((k, 1), sos, dtype=torch.long).to(device)
    seqs = k_prev_words
    top_k_scores = torch.zeros(k, 1).to(device)

    complete_seqs = list()
    complete_seqs_scores = list()

    step = 1
    for i in range(max_len):
        # print(f"src : {src.shape}, k_prev_words : {k_prev_words.shape}, src_mask : {src_mask.shape}, src_key_padding_mask : {src_key_padding_mask.shape}, memory_key_padding_mask : {memory_key_padding_mask.shape}")
        # input("PRESS ENTER TO CONTINUE.")
        tgt = model(src, k_prev_words, src_mask, None, src_key_padding_mask, None, memory_key_padding_mask)
        next_token_logits = tgt[:, -1, :]
        if step == 1:
            top_k_scores, top_k_words = next_token_logits[0].topk(k, dim=0, largest=True, sorted=True)
        else:
            top_k_scores, top_k_words = next_token_logits.view(-1).topk(k, 0, True, True)
        prev_word_inds = top_k_words / vocab_size
        next_word_inds = top_k_words % vocab_size

        prev_word_inds = prev_word_inds.long()

        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != eos]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])

        k -= len(complete_inds)

        if k == 0:
            break

        seqs = seqs[incomplete_inds]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        if step > max_len:
            break
        step += 1
    print(f"Complete seqs: {complete_seqs}")
    print(f"Complete seqs scores: {complete_seqs_scores}")
    input("PRESS ENTER TO CONTINUE.")


    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    print(seq)
    input("PRESS ENTER TO CONTINUE.")

    return seq

def temperature(logits, tem = 1):
    return logits / tem

def top_p(logits, p = 0.9):
    # 对送进来的log prob分布做排序
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    # 做softmax并累加
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    # 记下前top概率之外的词的index
    sorted_indices_to_remove = cumulative_probs > p
    # 
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    # 设置掩码，准备除去这些top之外词
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
    # 除去
    logits[indices_to_remove] = float('-inf')
    # print(logits)
    # print(logits.shape)
    # input("Press Enter to continue")
    return logits

def top_k(logits, k = 5):
    # 找出前k个最大的log prob
    _, top_k_indices = torch.topk(logits, k=k, dim=-1)
    # 设置掩码，准备除去这些top之外词
    top_k_indices = top_k_indices
    indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(dim=-1, index=top_k_indices, src=torch.ones_like(logits, dtype=torch.bool))
    # 除去
    logits[indices_to_remove] = float('-inf')
    # print(logits)
    # print(logits.shape)
    # input("Press Enter to continue")
    return logits

def predict(decoder_algorithm = "top_p"):
    model.eval()
    unfinished_sent_count = 0
    with torch.no_grad():
        with open(os.path.join(output_folder, f"predict.csv"), "w", encoding="utf-8") as f:
            df = pd.DataFrame(columns=["sent1", "sent2", "sent3", "sent4"])
            for data in tqdm(data_loader_test, dynamic_ncols=True, desc="Predicting"):
                src_batch, src_key_padding_mask = data[0].to(device), data[1].to(device)
                src_mask = gen_mask(src_batch.shape[1]).to(device)
                memory_key_padding_mask = src_key_padding_mask.clone()
                sentences, flag = decode(src_batch, src_key_padding_mask, max_len, decoder=decoder_algorithm, p = P, k = K, t = TEMPEARATURE)
                if flag == True:
                    unfinished_sent_count += 1
                # sentences = beam_search(src_batch, src_key_padding_mask, max_len)
                # print(sentences)

                sentences_str = []
                for sent in sentences:
                    out_str = ""
                    for i in range(1, sent.shape[1] - 1):
                        out_str += dataset.dictionary.tkn2word[sent[0][i].item()]
                        out_str += " "
                    sentences_str.append(out_str)
                while len(sentences_str) < 4:
                    sentences_str.append(" ")
                # print(sentences_str)
                # input("Press to continue..")
                df.loc[len(df.index)] = sentences_str
                # print(out_str)
            df.to_csv(f, index=False, header=True, encoding="utf-8")
            print(f"Total {len(df.index)} predictions, {unfinished_sent_count} of them unfinished.")
        
if __name__ == '__main__':
    dataset_folder = './data'
    output_folder = './output'

    device = torch.device("cuda:5") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    embedding_dim = 300  # 每个词向量的维度
    max_len = 128  # 每个句子预设的最大 token 数
    batch_size = 32
    DECODER = "top_p"     # "top_p" / "greedy" / "top_k" / "top_k_p" / "temp_only"
    P = 0.9                 # If you choose "greedy", "temp_only" or "top_k" as DECODER, param P wont function
    K = 20                   # If you choose "greedy", "temp_only" or "top_p" as DECODER, param K wont function
    TEMPEARATURE= 1.05       # If you choose "greedy" as DECODER, param TEMPERATURE wont function
    dataset = Corpus(dataset_folder, max_len, embedding_dim, need_vectorize = False)

    data_loader_train = DataLoader(dataset=dataset.train, batch_size=batch_size, shuffle=True)
    data_loader_valid = DataLoader(dataset=dataset.val, batch_size=batch_size, shuffle=False)
    data_loader_test = DataLoader(dataset=dataset.test, batch_size=1, shuffle=False)

    model = torch.load(os.path.join(output_folder, f"model_5.pkl"), map_location=device)
    predict(DECODER)
