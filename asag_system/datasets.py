import datasets
import torch
from torch.utils.data import Dataset

from asag_system.models import TripletTokenizer, label_map


class TripletClassificationDataset(Dataset):
    def __init__(self, data: datasets.DatasetDict, tokenizer: TripletTokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.data[idx]
        tokenized_input = self.tokenizer(
            item["reference_answer"], item["question"], item["student_answer"]
        ).squeeze()
        labels = torch.tensor(label_map[item["label_5way"]])
        return {
            "input_ids": tokenized_input,
            "labels": labels,
        }
