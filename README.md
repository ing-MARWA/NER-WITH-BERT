# Named Entity Recognition with Fine-tuned BERT

This repository contains a Python notebook that demonstrates how to fine-tune a BERT model for Named Entity Recognition (NER) using the Transformers library. The model is trained on the "Named Entity Recognition (NER) Corpus" dataset from Kaggle and can identify entities like person names, organizations, locations, and more.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Usage](#usage)
- [Model and Training](#model-and-training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [License](#license)

## Introduction

Named Entity Recognition (NER) is a crucial task in Natural Language Processing (NLP) where the goal is to identify and classify named entities in text. This project leverages the powerful BERT (Bidirectional Encoder Representations from Transformers) model, fine-tuning it to excel at NER.

## Dataset

- **Dataset Source:** [https://www.kaggle.com/datasets/naseralqaydeh/named-entity-recognition-ner-corpus](https://www.kaggle.com/datasets/naseralqaydeh/named-entity-recognition-ner-corpus)
- **Entities:** The dataset consists of sentences with labeled entities:
    - **geo:** Geographical Entity
    - **org:** Organization
    - **per:** Person
    - **gpe:** Geopolitical Entity
    - **tim:** Time indicator
    - **art:** Artifact
    - **eve:** Event
    - **nat:** Natural Phenomenon

## Requirements

- Python 3.7+
- Install the required libraries:
  ```bash
  pip install transformers evaluate seqeval wandb pandas scikit-learn
  ```

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repository.git
   ```
2. Download the dataset from the source mentioned above.
3. Update the dataset path in the notebook.
4. Run the Jupyter notebook:
   ```bash
   jupyter notebook ner_with_bert.ipynb
   ```

## Model and Training

- We utilize the `bert-base-uncased` model from the Transformers library.
- The model is fine-tuned specifically for token classification, predicting the entity tag for each token in a sentence.
- Training parameters like batch size, learning rate, and epochs are configurable.

## Inference

- The `infer_entities` function demonstrates how to use the fine-tuned model for predictions.
- It takes a sentence as input and outputs a list of (token, predicted_label) tuples.

## Evaluation

- The model's performance is evaluated using the `seqeval` library, which is suitable for sequence labeling tasks.
- Metrics like precision, recall, F1-score, and accuracy are calculated to assess the model's effectiveness.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
