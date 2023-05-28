# nlp-baselines-2022
The repository contains the tope performing NLP solutions for IGLU 2022 competition. The baselines cover solutions to **When** and **What** to ask as clarification questions. 
For more details about the baselines, we refer to [our paper](https://arxiv.org/pdf/2305.10783). 

| Task | Method                                 | Train | Trained Model | Test |
|------|----------------------------------------|-------|---------------|------|
| When | Text-Grid Cross Modularity             |       |               |      |
| When | Textual Grid world State               | [link](https://github.com/iglu-contest/nlp-baselines-2022/blob/main/classifiers/Textual%20Grid%20world%20State%20Baseline/train.ipynb)      |               |   [Link]((https://drive.google.com/drive/folders/11F_m8Qihv8AMZlfrr4P0-zrQOjPC8bnT?usp=drive_link)  | [Link](https://github.com/iglu-contest/nlp-baselines-2022/blob/main/classifiers/Textual%20Grid%20world%20State%20Baseline/test.py)
| What | Text-World Fusion Ranker               |       |               |      |
| What | State-Instruction Concatenation Ranker |       |               |      |

## When to ask
Here, we have two baselines which both predict if a given instruction is clear or if it needs a clarification question.:

1. **Text-Grid Cross Modularity**:

2. [**Textual Grid world State**]((https://github.com/iglu-contest/nlp-baselines-2022/tree/main/classifiers/Textual%20Grid%20world%20State%20Baseline)): You can use [train.ipynb]  to train the model yourself. In case you do not want to go through training, you can [download the trained models from here] and run [test.py] for inference purpose. 
 
## What to ask
1.  [**Text-World Fusion Ranker**]() This method utilizes a frozen DeBERTa-v3-base model to encode instructions and employs a text representation approach for ranking tasks. It combines the encoded text representation with a world representation derived from a 3D grid, which is processed through convolutional networks. The concatenated vector is passed through a two-layer MLP, and the model is trained using CrossEntropy loss with ensemble predictions from 10 models. Additionally, certain post-processing tricks are applied to enhance performance by excluding irrelevant questions based on heuristic rules.

3.  [**State-Instruction Concatenation Ranker**]() :This method focuses on aligning relevant queries and items closely in an embedding space while distancing queries from irrelevant items. It pairs positive questions with sampled negative questions and measures the similarity between the instruction and the question using a BERT-like language model. State information, such as colors and numbers of initialized blocks, is encoded as natural language and concatenated with the instruction. Data augmentation techniques, domain-adaptive fine-tuning, and the list-wise loss function are employed to improve training and generalization.
  -  Train
  -   Test
  -   Trained model

[work in progress]
