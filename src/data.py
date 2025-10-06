import os
import torch
import json
from datasets import Dataset, Features, Value, Sequence, Image, load_dataset

def load_custom_dataset(folder: str) -> Dataset:
    """
    Load dataset from a folder containing metadata.json and corresponding .png images.

    Args:
        folder (str): Path to the dataset folder.

    Returns:
        datasets.Dataset: HuggingFace Dataset with images and metadata.
    """
    with open(os.path.join(folder, "metadata.json"), "r") as f:
        meta = json.load(f)

    # convert dict-of-dicts into list-of-dicts with image paths
    rows = []
    for fname, md in meta.items():
        row = {"id": fname, "image": os.path.join(folder, fname + ".png")}
        row.update(md)
        rows.append(row)

    # define dataset schema
    features = Features({
        "id": Value("string"),
        "image": Image(),
        "gt": Value("string"),
        "linewidth": Value("int32"),
        "left": Value("int32"),
        "resolution": Value("int32"),
        "distances": Sequence(Value("float32")),
        "grid_size": Value("int32"),
    })

    dataset = Dataset.from_list(rows, features=features)

    dataset = dataset.map(lambda ex: {"label": ex["gt"]})

    return dataset


def load_lines_dataset_all(path="data/"):
    #generated train set (same params as test set from the paper)
    train_dataset = load_custom_dataset(f"{path}my2DlinePlots_train")
    train_dataset = train_dataset.filter(lambda ex: ex["resolution"] == 100)

    #test set from the paper
    paper_test_dataset = load_dataset("XAI/vlmsareblind")["valid"]
    paper_test_dataset = paper_test_dataset.map(lambda ex: {"label": ex["groundtruth"]})
    paper_test_dataset = paper_test_dataset.filter(lambda ex: ex['task'] == "Line Plot Intersections")
    paper_test_dataset = paper_test_dataset.map(
        lambda ex: {"meta": json.loads(ex["metadata"])},
        num_proc=4,
    )
    # test set with random colors
    color_test_dataset = load_custom_dataset(f"{path}my2DlinePlots_crandom")
    # test set with yellow and green lines
    cgy_test_dataset = load_custom_dataset(f"{path}my2DlinePlots_cgreenyellow")
    # test set with random colors and grid size 24
    grid_test_dataset = load_custom_dataset(f"{path}my2DlinePlots_crandom_grid24")
    # test set with random colors and width 6
    width_test_dataset = load_custom_dataset(f"{path}my2DlinePlots_crandom_width6")

    datasets = {}
    datasets["train"] = train_dataset
    datasets["paper_version"] = paper_test_dataset
    datasets["color"] = color_test_dataset
    datasets["cgy"] = cgy_test_dataset
    datasets["color_grid24"] = grid_test_dataset
    datasets["color_width6"] = width_test_dataset
    return datasets

class QwenChatDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, processor, system_message, query_text):
        self.ds = hf_dataset
        self.proc = processor
        self.system_msg = system_message
        self.user_msg = query_text

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        img = ex['image']
        label = ex['label']

        # Build prompt
        system_prompt = f"<|im_start|>system\n{self.system_msg}\n<|im_end|>\n"
        user_prompt = (
            "<|im_start|>user\n"
            "<|vision_start|><|image_pad|><|vision_end|>"
            f"{self.user_msg}\n"
            "<|im_end|>\n"
        )
        assistant_prompt = "<|im_start|>assistant\n"
        prompt = system_prompt + user_prompt + assistant_prompt

        # Process text+image
        out = self.proc(
            text=prompt,
            images=img,
            return_tensors="pt",
            truncation=True,
        )
        input_ids = out.input_ids.squeeze(0)
        attention_mask = out.attention_mask.squeeze(0)
        pixel_values = out.pixel_values.squeeze(0)
        image_grid_thw = out.image_grid_thw.squeeze(0)

        # Tokenize label
        target = label + self.proc.tokenizer.eos_token
        target_ids = self.proc.tokenizer(
            target,
            return_tensors="pt"
        ).input_ids.squeeze(0)

        labels = torch.full(
            (input_ids.size(0) + target_ids.size(0),),
            -100,
            dtype=torch.long,
        )
        labels[input_ids.size(0):] = target_ids

        input_ids = torch.cat([input_ids, target_ids], dim=0)
        attention_mask = torch.cat([
            attention_mask,
            torch.ones_like(target_ids)
        ], dim=0)

        labels_nb = ex['label_nb']

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values,
            'image_grid_thw': image_grid_thw,
            'labels': labels,
            'labels_nb': labels_nb,
        }


def _left_pad(seqs, pad_value):
    length = max(s.size(0) for s in seqs)
    out = torch.full((len(seqs), length), pad_value, dtype=seqs[0].dtype, device=seqs[0].device)
    for i, s in enumerate(seqs):
        out[i, -s.size(0):] = s
    return out

class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        input_ids = _left_pad([b["input_ids"] for b in batch], self.pad_id)
        attention_mask = _left_pad([b["attention_mask"] for b in batch], 0)
        labels = _left_pad([b["labels"] for b in batch], -100)
        pixel_values = torch.stack([b["pixel_values"] for b in batch], dim=0)
        image_grid_thw = torch.stack([b["image_grid_thw"] for b in batch], dim=0)
        labels_nb = torch.tensor([b["labels_nb"] for b in batch], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "labels": labels,
            "labels_nb": labels_nb,
        }
