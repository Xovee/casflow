"""Parquet cascades, fixed dataset splits, and variable-length tensor batches."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import hashlib
import math
import random

import pyarrow.parquet as pq
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


SETTINGS = {
    "weibo": ((1800, 3600), 86400),
    "twitter": ((86400, 172800), 2764800),
    "aps": ((1095, 1826), 7305),
}
CHINA_TIME = timezone(timedelta(hours=8))


@dataclass
class Cascade:
    id: str
    events: list
    label: int


def read_cascades(raw, dataset, observation):
    prediction = SETTINGS[dataset][1]
    cascades = []
    with Path(raw).open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    batches = pq.ParquetFile(raw).iter_batches(
        batch_size=1024, columns=["cascade_id", "published", "events"], use_threads=False
    )
    for batch in batches:
        for record in batch.to_pylist():
            cascade_id, published = record["cascade_id"], record["published"]
            if dataset == "aps":
                # The public split excludes papers published during 1997.
                if int(published[:4]) >= 1997:
                    continue
            else:
                timestamp = datetime.fromtimestamp(float(published), CHINA_TIME)
                if dataset == "weibo":
                    # Retain the public TF split: the 1h setting uses 19:00
                    # to match historical DeepHawkes/CasCN preprocessing.
                    end_hour = 19 if observation == 3600 else 18
                    if not 8 <= timestamp.hour < end_hour:
                        continue
                elif timestamp.month == 4 and timestamp.day > 10:
                    continue
            events, future = [], 0
            for event in record["events"]:
                arrived = event["elapsed"]
                if arrived < observation:
                    events.append((tuple(event["path"]), arrived))
                if observation <= arrived < prediction:
                    future += 1
            if len(events) >= 10:
                events.sort(key=lambda event: event[1])
                cascades.append(Cascade(cascade_id, events, future))
    return cascades, digest


def split_cascades(cascades, seed=0):
    indices = list(range(len(cascades)))
    random.Random(seed).shuffle(indices)
    first, second = math.ceil(0.7 * len(indices)), math.ceil(0.85 * len(indices))
    membership = {}
    subsets = (indices[:first], indices[first:second], indices[second:])
    for name, subset in zip(("train", "val", "test"), subsets):
        membership.update((index, name) for index in subset)
    splits = {name: [] for name in ("train", "val", "test")}
    for index, cascade in enumerate(cascades):
        splits[membership[index]].append(cascade)
    return splits


class Cascades(Dataset):
    def __init__(self, tensors):
        self.features = tensors["features"]
        self.labels = tensors["labels"]
        self.lengths = tensors["lengths"]
        self.offsets = torch.cat((torch.zeros(1, dtype=torch.long), self.lengths.cumsum(0)))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.features[self.offsets[index]:self.offsets[index + 1]], self.labels[index]


def collate(batch):
    features, labels = zip(*batch)
    lengths = torch.tensor([len(x) for x in features])
    return pad_sequence(features, batch_first=True), lengths, torch.stack(labels)
