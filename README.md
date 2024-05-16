
# Overview and Setup

This project uses Poetry for dependency management.

https://python-poetry.org/docs/#installing-with-the-official-installer

The important parts of this project are:
- `asag_system/` - A python package containing the core codebase used for this task.
- `workspace/eda.ipynb` - A Jupyter notebook performing EDA on the dataset.
- `workspace/baseline.ipynb` - A notebook to get a sense of task difficulty using a baseline model that predicts the most frequent class.
- `workspace/train.ipynb` - The main notebook used for modelling and experimentation.
- `workspace/test.ipynb` - A final notebook evaluating the models on the test set.

The notebooks should work when run in the above order.

# Objectives

I aimed to build a robust MVP for Automatic Short Answer Grading (ASAG) in the allotted 4 hours. The system should provide useful feedback on student answers to short answer questions. Given the timeframe, I focus on approaches that work well out of the box but also have potential for extension and improvement.

Based on the dataset labels, I framed the task as a 5-class classification problem. The test set consists of two sections: unseen answers and unseen questions. In the unseen answers section, the model knows the question and reference answer, but the student answers are new. In the unseen questions section, both questions and reference answers are unknown to the model.


# Methodology

The dataset is already split into a train and test set. I further divided the train set (renamed as the development set) into training and validation sets with an 80:20 ratio. During the experimentation phase, I trained models on the training set and evaluated on the validation set. After manual experimentation, I chose the best performing settings and retrained the model on the entire development set. This final model was then evaluated on the test set.


## Baseline Model

A model that simply classifies each student answer as "correct", the most frequent class, is used to establish a baseline for the task.

## DistilBERT Sentence Triplet Classifier

The main approach I used relies on text representations from a Large Language Model (LLM). Grading short answer questions requires identifying student answers that are phrased differently to the reference answer but contain the same information. This is complicated by things such as synonyms and grammar rules in natural language. Pre-trained LLMs are very useful in this case as they have already learned the generic rules of language along with the relationship between words and allow our training to focus on our specific task.

It has been noted that student answers often omit information from the question, while reference answers repeat it [1]. Including the question as a model input not only gives additional context but also helps to more accurately evaluate the student's answer in these cases.

Given these two points, a natural way to frame this is to extend BERT's Sentence Pair Classification task into a Sentence Triplet Classification task. Model inputs are constructed as a string with the following structure:

```
[CLS] {reference_answer} [SEP] {question} [SEP] {student_answer} [SEP]
```

BERT style language models learn that the `[SEP]` token acts as a separator between two segments during pre-training. I leverage this knowledge and extend it for my new task. I use DistilBERT which aims to perform similarly to BERT but is smaller and more practical to train on consumer hardware. The input string is tokenised, embedded, and passed through the encoder's self-attention layers. The encoded classification token is used as the sequence level representation and is passed into a classification head. The classification head's 5 output nodes represent the class logits.

## Pre-processing and Training Details

I trained this model in batches of 16. Each item was padded to the length of the longest sequence in its batch. An attention mask was used to index these pad tokens and they are ignored in the self-attention layers. I trained this model for 3 epochs.

## Evaluation Metrics

Two metrics were used for evaluation. Accuracy is straightforward to understand, representing the proportion of student answers correctly labelled. As accuracy can be misleading when classes are imbalanced I also use Macro F1 Score, which is the mean of the F1 Scores across all classes.


# Results

|                               | Unseen Answers |          | Unseen Questions |          |
|-------------------------------|----------------|----------|------------------|----------|
|                               | Accuracy       | Macro F1 | Accuracy         | Macro F1 |
| Most Frequent Baseline        | 0.40           | 0.12     | 0.40             | 0.12     |
| DistilBERT Triplet Classifier | 0.73           | 0.66     | 0.65             | 0.62     |

The DistilBERT classifier significantly outperforms the baseline across both portions of the test set. Unsurprisingly, this model performs better on the Unseen Answers portion compared to the Unseen Questions, although the difference is fairly small. This suggests that the model generalises fairly well to new questions in the same domain.

As noted above, Accuracy can be misleading as naive classifiers such as our baseline can perform fairly well on the metric without providing any real value in practise. The Macro F1 Score instead represents a better measure of the usefulness of these model.


# Conclusion

Overall, this was a very interesting task. My solution follows a pragmatic approach, framing the problem in a natural way and leveraging openly available pre-trained models to arrive at a good solution. Its strengths are in its simplicity and extensibility. Below I outline potential improvements for the model.

## Weaknesses and Potential Improvements

Given the limited time, I opted for a simple training and validation set split during model experimentation. Experimenting using cross-validation would result in choosing model architectures and hyperparameters that would generalise better to unseen data.

Another limitation was the lack of incorporation of the tabular features such as "reference_answer_quality". One potential way to explore this would be to embed these tabular features into a "context embedding" which is then concatenated to the sequence embedding currently used.

When performing sentence pair classification, BERT adds a segment embedding to each token representation to indicate whether or not it belongs to the first or second sentence. In this implementation I rely only on the [SEP] token to separate the three sentences. Extending BERT's segment embeddings to allow for more than two segments could allow the model to better learn the relationship between the three sentences.

# References

[1] Dzikovska, M. O., Nielsen, R., Brew, C., Leacock, C., Giampiccolo, D., Bentivogli, L., ... & Dang, H. T. (2013, June). Semeval-2013 task 7: The joint student response analysis and 8th recognizing textual entailment challenge