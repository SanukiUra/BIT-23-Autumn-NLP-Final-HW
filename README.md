# BIT-23-Autumn-NLP-Final-HW

🏫 Beijing Institute of Technology

📚 2023 Autumn NLP Course Assignment

✍️ by Sanuki Ura 宇良讚歧

## Assignment Requirements

1. Story Generation
   Use Dataset [ROCStories](https://cs.rochester.edu/nlp/rocstories/) for generation. Given the first sentence and generate the next four sentences.

   Choose your own architecture
   You can use LSTM/Transformer/CNN layers of PyTorch lib.

   Implement at least one decoding algorithm
   You can use Beam Search/Top-p Sampling/Top-k Sampling/Temperature Process as your decoding algorithm

   Generate correct sequences
   Use \<BOS> and \<EOS> or something to control.

2. Fine-Tuning LLMs
   Use an LLM (GPT2/LLaMA/...)  and fine-tune it using suitable methods. 

3. Evaluation Your Model

   Natural language generation itself is not easy to evaluate. You can choose one or more evaluation methods to analyze the model.

> This repository only includes task1 and task3. Using HuggingFace libs you can easily fine-tuning LLMs for task2.

## Requirements

Python>=3.7

Make sure you can use PyTorch.

```python
pip install -r requirements.txt
```

## Usage

### Training

Modify the parameters in `Train.py` if you need.
Then,

```python
python Train.py
```

### Inferencing

Make sure there are some '.pkl' models in folder `./output`,
choose your model in line 239 of `Test.py`, 
modify some parameters if you need.
Then,

```python
python Test.py
```

### Evaluating

Use BLEU benchmark to evaluate the generated texts.
Make sure there are output files like `top_p.csv` or `top_k.csv` in folder `./output`, then modify the 'id' in the file `Eval.py` to the corresponding file name.
Then,

```python
python Eval.py
```

## Results

You can see an example result in file `./ouput/top_p.csv`.
