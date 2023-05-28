# nlp-baselines-2022
The repository contains the tope performing NLP solutions for IGLU 2022 competition. The baselines cover solutions to **When** and **What** to ask as clarification questions. 
For more details about the baselines, we refer to [our paper](https://arxiv.org/pdf/2305.10783). 

| Task | Method                                 | Train | Trained Model | Test |
|------|----------------------------------------|-------|---------------|------|
| When | Text-Grid Cross Modularity             |       |               |      |
| When | Textual Grid world State     | [link](https://github.com/iglu-contest/nlp-baselines-2022/blob/main/classifiers/Textual%20Grid%20world%20State%20Baseline/train.ipynb)      |   [Link](https://drive.google.com/drive/folders/11F_m8Qihv8AMZlfrr4P0-zrQOjPC8bnT?usp=drive_link)  | [Link](https://github.com/iglu-contest/nlp-baselines-2022/blob/main/classifiers/Textual%20Grid%20world%20State%20Baseline/test.py)
| What | Text-World Fusion Ranker               |       |               |      |
| What | State-Instruction Concatenation Ranker |       |               |      |

## When to ask
Here, we have two baselines which both predict if a given instruction is clear or if it needs a clarification question.:

1. **Text-Grid Cross Modularity**: This baseline consists of four major components. First, the utterance encoder represents dialogue utterances by adding Architect and Builder annotations before each respective utterance and encoding them using pre-trained language models. Second, the world state encoder represents the pre-built structure using a voxel-based grid and applies convolutional layers to capture spatial dependencies and abstract representations. Third, the fusion module combines the world state and dialogue history representations using self-attention and cross-attention layers. Fourth, the slot decoder performs linear projection and binary classification to obtain the final output. This approach has shown improved performance compared to the simple LLM fine-tuning approach.

2. [**Textual Grid world State**]((https://github.com/iglu-contest/nlp-baselines-2022/tree/main/classifiers/Textual%20Grid%20world%20State%20Baseline)): This baseline enhances the classification task by mapping the GridWorld state to a textual context, which is then combined with the verbalizations of the Architect-Agent. It utilizes an automated description of the number of blocks per color in the pre-built structures to provide valuable contextual information. By incorporating this textual description, the baseline achieves a 4% improvement in performance, highlighting the significance of relevant contextual information for better understanding and classification in language-guided collaborative tasks.


## What to ask
1.  [**Text-World Fusion Ranker**]() This method utilizes a frozen DeBERTa-v3-base model to encode instructions and employs a text representation approach for ranking tasks. It combines the encoded text representation with a world representation derived from a 3D grid, which is processed through convolutional networks. The concatenated vector is passed through a two-layer MLP, and the model is trained using CrossEntropy loss with ensemble predictions from 10 models. Additionally, certain post-processing tricks are applied to enhance performance by excluding irrelevant questions based on heuristic rules.

3.  [**State-Instruction Concatenation Ranker**]() :This method focuses on aligning relevant queries and items closely in an embedding space while distancing queries from irrelevant items. It pairs positive questions with sampled negative questions and measures the similarity between the instruction and the question using a BERT-like language model. State information, such as colors and numbers of initialized blocks, is encoded as natural language and concatenated with the instruction. Data augmentation techniques, domain-adaptive fine-tuning, and the list-wise loss function are employed to improve training and generalization.
