# nlp-baselines-2022
The repository contains the tope performing NLP solutions for IGLU 2022 competition. The baselines cover solutions to **When** and **What** to ask as clarification questions. 
For more details about the baselines, we refer to [our paper](https://arxiv.org/pdf/2305.10783). 

| Task | Method                                 | Train | Trained Model | Test |
|------|----------------------------------------|-------|---------------|------|
| When | Bert-based Classifier          |  [Link](https://colab.research.google.com/drive/1k4dbDcKxR6ct1zrhQ2bUo69Zq7FpSGxO#scrollTo=JYFBB8pe1QSR)   | [Link](https://colab.research.google.com/drive/1k4dbDcKxR6ct1zrhQ2bUo69Zq7FpSGxO#scrollTo=ZZoC_BXxgAsU) | [Link](https://colab.research.google.com/drive/1k4dbDcKxR6ct1zrhQ2bUo69Zq7FpSGxO#scrollTo=gAg6iYd4w2qL)|
| When | Text-Grid Cross Modularity             |  [Link](https://github.com/iglu-contest/nlp-baselines-2022/blob/main/classifiers/Text-Grid%20Cross%20Modularity/train.py)     |  [Link](https://drive.google.com/drive/folders/18zffowv6RVzrw9UywvyQe2-YysEp4nf3?usp=sharing)       | [Link](https://github.com/iglu-contest/nlp-baselines-2022/blob/main/classifiers/Text-Grid%20Cross%20Modularity/test.py)     |
| When | Textual Grid world State     | [Link](https://github.com/iglu-contest/nlp-baselines-2022/blob/main/classifiers/Textual%20Grid%20world%20State%20Baseline/train.ipynb)      |   [Link](https://drive.google.com/drive/folders/11F_m8Qihv8AMZlfrr4P0-zrQOjPC8bnT?usp=drive_link)  | [Link](https://github.com/iglu-contest/nlp-baselines-2022/blob/main/classifiers/Textual%20Grid%20world%20State%20Baseline/test.py)
| What |BM25 Ranker  |  - | - | [Link](https://colab.research.google.com/drive/1k4dbDcKxR6ct1zrhQ2bUo69Zq7FpSGxO#scrollTo=GWy7a61l5PLS) |
| What | Text-World Fusion Ranker             |  [Link](https://github.com/iglu-contest/nlp-baselines-2022/blob/main/rankers/Text%20World%20Fusion%20Ranker/train.ipynb) |     [Link](https://drive.google.com/drive/folders/1qtGk5HxikNQFlalU7KqCyo5j7XS1lQFM?usp=drive_link)            | [Link](https://github.com/iglu-contest/nlp-baselines-2022/blob/main/rankers/Text%20World%20Fusion%20Ranker/test.py)        |
| What | State-Instruction Concatenation Ranker | [Link](https://github.com/iglu-contest/nlp-baselines-2022/blob/main/rankers/State-Instruction%20Concatenation%20Ranker/train.py)      | [Link](https://drive.google.com/drive/folders/1CvxrnACZz5O6z9XecSi7nPF0VsOvuXo2?usp=sharing) | [Link](https://github.com/iglu-contest/nlp-baselines-2022/blob/main/rankers/State-Instruction%20Concatenation%20Ranker/test.py)     |

## When to ask Clarifying questions
Here, we have two baselines which both predict if a given instruction is clear or if it needs a clarification question.:

1.  [**BERT-based Classifier**](https://colab.research.google.com/drive/1k4dbDcKxR6ct1zrhQ2bUo69Zq7FpSGxO?usp=sharing): . A straightforward baseline for identifying whether an instruction needs a clarifying question would involve fine-tuning language models like BERT. A classification layer is be added to predict the clarity of the instructions. 

2.  [**Text-Grid Cross Modularity**](https://github.com/iglu-contest/nlp-baselines-2022/tree/main/classifiers/Text-Grid%20Cross%20Modularity): This baseline consists of four major components. First, the utterance encoder represents dialogue utterances by adding Architect and Builder annotations before each respective utterance and encoding them using pre-trained language models. Second, the world state encoder represents the pre-built structure using a voxel-based grid and applies convolutional layers to capture spatial dependencies and abstract representations. Third, the fusion module combines the world state and dialogue history representations using self-attention and cross-attention layers. Fourth, the slot decoder performs linear projection and binary classification to obtain the final output. This approach has shown improved performance compared to the simple LLM fine-tuning approach.
3.  [**Textual Grid world State**](https://github.com/iglu-contest/nlp-baselines-2022/tree/main/classifiers/Textual%20Grid%20world%20State%20Baseline): This baseline enhances the classification task by mapping the GridWorld state to a textual context, which is then combined with the verbalizations of the Architect-Agent. It utilizes an automated description of the number of blocks per color in the pre-built structures to provide valuable contextual information. By incorporating this textual description, the baseline achieves a 4% improvement in performance, highlighting the significance of relevant contextual information for better understanding and classification in language-guided collaborative tasks.


## What to ask as Clarifying questions
1.    [**BM25**](https://colab.research.google.com/drive/1k4dbDcKxR6ct1zrhQ2bUo69Zq7FpSGxO#scrollTo=GWy7a61l5PLS): The task is formulated  as ranking a pool of clarifying questions based on their relevance to ambiguous instructions to place the most pertinent clarifying questions at the top of the ranked list. BM25 as one of the most-well known and widely used ranker is employed as one of the most simple baselines for this task.
2.    [**Text-World Fusion Ranker**](https://github.com/iglu-contest/nlp-baselines-2022/tree/main/rankers/Text%20World%20Fusion%20Ranker): This method utilizes a frozen DeBERTa-v3-base model to encode instructions and employs a text representation approach for ranking tasks. It combines the encoded text representation with a world representation derived from a 3D grid, which is processed through convolutional networks. The concatenated vector is passed through a two-layer MLP, and the model is trained using CrossEntropy loss with ensemble predictions from 10 models. Additionally, certain post-processing tricks are applied to enhance performance by excluding irrelevant questions based on heuristic rules.
3.    [**State-Instruction Concatenation Ranker**](https://github.com/iglu-contest/nlp-baselines-2022/tree/main/rankers/State-Instruction%20Concatenation%20Ranker) :This method focuses on aligning relevant queries and items closely in an embedding space while distancing queries from irrelevant items. It pairs positive questions with sampled negative questions and measures the similarity between the instruction and the question using a BERT-like language model. State information, such as colors and numbers of initialized blocks, is encoded as natural language and concatenated with the instruction. Data augmentation techniques, domain-adaptive fine-tuning, and the list-wise loss function are employed to improve training and generalization.


# Related papers
[Transforming Human-Centered AI Collaboration: Redefining Embodied Agents Capabilities through Interactive Grounded Language Instructions](https://arxiv.org/abs/2305.10783)
```
@misc{mohanty2023transforming,
      title={Transforming Human-Centered AI Collaboration: Redefining Embodied Agents Capabilities through Interactive Grounded Language Instructions}, 
      author={Shrestha Mohanty and Negar Arabzadeh and Julia Kiseleva and Artem Zholus and Milagro Teruel and Ahmed Awadallah and Yuxuan Sun and Kavya Srinet and Arthur Szlam},
      year={2023},
      eprint={2305.10783},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```

[IGLU 2022: Interactive Grounded Language Understanding in a Collaborative Environment at NeurIPS 2022](https://arxiv.org/abs/2205.13771)
```
@misc{kiseleva2022iglu,
      title={IGLU 2022: Interactive Grounded Language Understanding in a Collaborative Environment at NeurIPS 2022}, 
      author={Julia Kiseleva and Alexey Skrynnik and Artem Zholus and Shrestha Mohanty and Negar Arabzadeh and Marc-Alexandre Côté and Mohammad Aliannejadi and Milagro Teruel and Ziming Li and Mikhail Burtsev and Maartje ter Hoeve and Zoya Volovikova and Aleksandr Panov and Yuxuan Sun and Kavya Srinet and Arthur Szlam and Ahmed Awadallah},
      year={2022},
      eprint={2205.13771},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
[Collecting Interactive Multi-modal Datasets for Grounded Language Understanding](https://arxiv.org/abs/2211.06552)
```
@misc{mohanty2023collecting,
      title={Collecting Interactive Multi-modal Datasets for Grounded Language Understanding}, 
      author={Shrestha Mohanty and Negar Arabzadeh and Milagro Teruel and Yuxuan Sun and Artem Zholus and Alexey Skrynnik and Mikhail Burtsev and Kavya Srinet and Aleksandr Panov and Arthur Szlam and Marc-Alexandre Côté and Julia Kiseleva},
      year={2023},
      eprint={2211.06552},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

[Learning to Solve Voxel Building Embodied Tasks from Pixels and Natural Language Instructions](https://arxiv.org/abs/2211.00688)
```
@misc{skrynnik2022learning,
      title={Learning to Solve Voxel Building Embodied Tasks from Pixels and Natural Language Instructions}, 
      author={Alexey Skrynnik and Zoya Volovikova and Marc-Alexandre Côté and Anton Voronov and Artem Zholus and Negar Arabzadeh and Shrestha Mohanty and Milagro Teruel and Ahmed Awadallah and Aleksandr Panov and Mikhail Burtsev and Julia Kiseleva},
      year={2022},
      eprint={2211.00688},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```
