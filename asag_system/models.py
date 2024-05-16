from typing import Callable

import torch
from sklearn.metrics import f1_score, accuracy_score
from torch import nn
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    DistilBertConfig,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.trainer_utils import EvalPrediction

TripletTokenizer = Callable[[str, str, str], torch.Tensor]

label_map = {
    "correct": 0,
    "partially_correct_incomplete": 1,
    "contradictory": 2,
    "irrelevant": 3,
    "non_domain": 4,
}


def compute_metrics(eval_pred: EvalPrediction):
    logits, labels = eval_pred
    predictions = logits.argmax(-1)

    f1 = f1_score(predictions, labels, average="macro")
    accuracy = accuracy_score(predictions, labels)

    return {"macro_f1": f1, "accuracy": accuracy}


class DistilBertTripletTokenizer:
    def __init__(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    def __call__(self, sentence1: str, sentence2: str, sentence3: str) -> torch.Tensor:
        concatenated = f"[CLS] {sentence1} [SEP] {sentence2} [SEP] {sentence3} [SEP]"
        return self.tokenizer.encode(
            concatenated,
            add_special_tokens=False,
            return_tensors="pt",
        )


class SentenceTripletClassifier(nn.Module):
    def __init__(self, num_labels: int = 5):
        super().__init__()
        self.config = DistilBertConfig.from_pretrained(
            "distilbert-base-uncased", num_labels=num_labels
        )
        self.distilbert = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", config=self.config
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> SequenceClassifierOutput:
        return self.distilbert(
            input_ids=input_ids, labels=labels, attention_mask=attention_mask
        )


class MostFrequentBaseline(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, input_ids: torch.Tensor, labels: torch.Tensor, **kwargs
    ) -> SequenceClassifierOutput:
        """Always returns class 0"""
        batch_size = input_ids.shape[0]
        logits = torch.zeros(batch_size, len(label_map), device=input_ids.device)
        logits[:, 0] = 1
        loss = nn.CrossEntropyLoss()(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)
