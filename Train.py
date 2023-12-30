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

def validate():
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    total_loss = []
    with torch.no_grad():
        for data in tqdm(data_loader_valid, dynamic_ncols=True, desc="Validating"):
            src_batch, tgt_batch, src_key_padding_mask, tgt_key_padding_mask = data[0].to(device), data[1].to(
                device), data[2].to(device), data[3].to(device)
            src_mask = gen_mask(src_batch.shape[1]).to(device)
            memory_key_padding_mask = src_key_padding_mask.clone()
            tgt_inputs, tgt_outputs = tgt_batch[:, :-1].to(device), tgt_batch[:, 1:].to(device)

            tgt_mask = gen_mask(tgt_inputs.shape[1]).to(device)
            tgt_key_padding_mask = tgt_key_padding_mask[:, :-1]

            y_hat = model(src_batch, tgt_inputs, src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask,
                          memory_key_padding_mask)

            tgt_outputs = tgt_outputs.to(torch.int64)
            loss = criterion(y_hat.contiguous().view(-1, y_hat.shape[-1]), tgt_outputs.contiguous().view(-1))
            total_loss.append(loss.item())
        validate_loss = sum(total_loss) / len(total_loss)
    print(f"Validate loss: {validate_loss}")

    return validate_loss

def train(model):
    pass
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()

        total_loss = []

        tqdm_iterator = tqdm(data_loader_train, dynamic_ncols=True, desc=f'Epoch {epoch + 1}/{num_epochs}')

        for data in tqdm_iterator:
            batch_x, batch_y, x_padding_masks, y_padding_masks = data[0].to(device), data[1].to(device), data[2].to(
                device), data[3].to(device)
            src_mask = gen_mask(batch_x.shape[1]).to(device)


            tgt_inputs, tgt_outputs = batch_y[:, :-1].to(device), batch_y[:, 1:].to(device)
            tgt_mask = gen_mask(tgt_inputs.shape[1]).to(device)

            memory_key_padding_mask = x_padding_masks.clone()
            tgt_key_padding_mask = y_padding_masks[:, :-1]
            y_hat = model(batch_x, tgt_inputs, src_mask, tgt_mask, x_padding_masks, tgt_key_padding_mask,
                          memory_key_padding_mask)

            tgt_outputs = tgt_outputs.to(torch.int64)
            # print(f"y_hat: {y_hat.shape}, label: {tgt_outputs.shape}")
            # a = y_hat.contiguous().view(-1, y_hat.shape[-1])
            # b = tgt_outputs.contiguous().view(-1)
            # print(f"y_hat 张开: {a.shape}, label 张开： {b.shape}")
            # input("Press Enter to fuck me")
            loss = criterion(y_hat.contiguous().view(-1, y_hat.shape[-1]), tgt_outputs.contiguous().view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.append(loss.item())

            tqdm_iterator.set_postfix(loss=sum(total_loss) / len(total_loss))

        tqdm_iterator.close()

        train_loss = sum(total_loss) / len(total_loss)

        validate_loss = validate()

        print(f"Epoch {epoch + 1}/{num_epochs} train_loss: {train_loss}")
        torch.save(model, os.path.join(output_folder, f"model_{epoch}.pkl"))

    return model


if __name__ == '__main__':
    dataset_folder = './data'
    output_folder = './output'

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    embedding_dim = 300  # 每个词向量的维度
    max_len = 128  # 每个句子预设的最大 token 数
    batch_size = 32
    num_epochs = 10
    lr = 1e-4
    # 可选使用词向量和不使用词向量，我用的是fasttext的wiki-news-300d-1M.vec
    dataset = Corpus(dataset_folder, max_len, embedding_dim, need_vectorize = False)

    data_loader_train = DataLoader(dataset=dataset.train, batch_size=batch_size, shuffle=True)
    data_loader_valid = DataLoader(dataset=dataset.val, batch_size=batch_size, shuffle=False)
    data_loader_test = DataLoader(dataset=dataset.test, batch_size=1, shuffle=False)
    print(dataset.vocab_size)
    input("Press")

    model = Transformer(max_len=max_len, vocab_size=dataset.vocab_size, embedding_weight=dataset.embedding_weight).to(
        device)

    model = train(model)