from tarfile import TruncatedHeaderError

import torch
import os
from torchvision import transforms
from datasets import load_dataset
from transformers import AutoImageProcessor
from torchvision.transforms import (
        CenterCrop,
        Compose,
        Normalize,
        RandomHorizontalFlip,
        RandomResizedCrop,
        Resize,
        ToTensor,
    )
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence


seed = 42

@dataclass
class DataArguments:
    eval_dataset_size: int = field(
        default=1024, metadata={"help": "Size of validation dataset."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    source_max_len: int = field(
        default=1024,
        metadata={"help": "Maximum source sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    target_max_len: int = field(
        default=256,
        metadata={"help": "Maximum target sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    dataset: str = field(
        default='alpaca',
        metadata={"help": "Which dataset to finetune on. See datamodule for options."}
    )
    dataset_format: Optional[str] = field(
        default=None,
        metadata={"help": "Which dataset format is used. [alpaca|chip2|self-instruct|hh-rlhf]"}
    )


def collate_fn(examples):
    """Custom collate function to handle variable batch sizes."""
    pixel_values = [example["pixel_values"] for example in examples]
    labels = torch.tensor([example["fine_label"] for example in examples])  # for cifar100

    # Stack pixel_values into a tensor, handling cases with fewer samples than batch size
    pixel_values = torch.stack(pixel_values) if len(pixel_values) > 0 else torch.empty(0)

    return {"pixel_values": pixel_values, "labels": labels}

def create_data_loader(dataset_name, batch_size, model_checkpoint):#, debug_flag=False):
    """Create a llm loader for the specified dataset."""
    if torch.cuda.is_available():
        percentage_of_data = 1 # 0.02 #0.02 # just for debugging purposes
    else:
        percentage_of_data = 1
    percentage_of_data = 1 
    percentage_of_data_test = 1

    if dataset_name.lower() in ["food101", "imagenet-1k"]:
        split_test = "validation" # test dataset has no label in imagenet-1k
    elif dataset_name.lower() == "zh-plus/tiny-imagenet":
        split_test = "valid"
    else:
        split_test = "test"


    dataset = load_dataset(dataset_name)
    total_train_samples = len(dataset["train"])
    total_test_samples = len(dataset[split_test])
    num_samples_train = int(total_train_samples * percentage_of_data)
    num_samples_test = int(total_test_samples * percentage_of_data_test)

    dataset_shuffled_train = dataset["train"].shuffle(seed=seed)
    dataset_train = dataset_shuffled_train.select(range(num_samples_train))
    dataset_shuffled_test = dataset[split_test].shuffle(seed=seed)
    dataset_test = dataset_shuffled_test.select(range(num_samples_test))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    image_column = "image" if "image" in dataset_train.column_names else "img"
    if dataset_name.lower() in ["food101", "caltech101", "imagenet-1k"]:
        image_column = "image"

    train_dataset = dataset_train.with_transform(lambda x: {**x, 'pixel_values': transform(x[image_column])})
    test_dataset = dataset_test.with_transform(lambda x: {**x, 'pixel_values': transform(x[image_column])})

    # Create DataLoader with custom collate_fn
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model_checkpoint = "google/vit-base-patch16-224-in21k"

    image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)

    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    train_transforms = Compose(
        [
            RandomResizedCrop(image_processor.size["height"]),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )

    val_transforms = Compose(
        [
            Resize(image_processor.size["height"]),
            CenterCrop(image_processor.size["height"]),
            ToTensor(),
            normalize,
        ]
    )

    def preprocess_train(example_batch):
        """Apply train_transforms across a batch."""
        # example_batch["pixel_values"] = [train_transforms(image.convert("RGB")) for image in example_batch["image"]]
        #
        # example_batch["pixel_values"] = [train_transforms(img.convert("RGB")) for img in
        #                                  example_batch["img"]]  # for cifar100
        example_batch["pixel_values"] = [train_transforms(img.convert("RGB")) for img in example_batch[image_column]]
        return example_batch

    def preprocess_val(example_batch):
        """Apply val_transforms across a batch."""
        # example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
        # example_batch["pixel_values"] = [val_transforms(img.convert("RGB")) for img in
        #                                  example_batch["img"]]  # for cifar100
        example_batch["pixel_values"] = [val_transforms(img.convert("RGB")) for img in example_batch[image_column]]

        return example_batch

    dataset = {
        "train": dataset_train,
        "test": dataset_test
    }

    # Print dataset sizes
    print(f"Original training dataset size: {total_train_samples}")
    print(f"Original test dataset size: {total_test_samples}")
    print(f"Size of training subset: {len(dataset_train)}")
    print(f"Size of test subset: {len(dataset_test)}")

    if "fine_label" in dataset["train"].features:
        label_column = "fine_label"
    else:
        label_column = "label"

    label_names = dataset["train"].features[label_column].names
    label2id, id2label = dict(), dict()
    for i, label_name in enumerate(label_names):
        label2id[label_name] = i
        id2label[i] = label_name
    # fine_label = dataset["train"].features["fine_label"].names
    # label2id, id2label = dict(), dict()
    # for i, fine_label in enumerate(fine_label):  # for cifar100, it changed from label to fine_label
    #     label2id[fine_label] = i
    #     id2label[i] = fine_label

    train_ds = dataset["train"]
    val_ds = dataset["test"]

    train_ds.set_transform(preprocess_train)
    val_ds.set_transform(preprocess_val)

    return train_loader, test_loader, train_ds, val_ds, label2id, id2label