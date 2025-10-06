import argparse
import logging
import random
import shutil, uuid
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from peft import (
    PeftModel,
)
from torch.utils.data import random_split

from src.data import load_lines_dataset_all, Collator, QwenChatDataset
from src.model import configure_trainable_params, load_model
from src.seed import set_seed
from qwen_vl_utils import process_vision_info
from src.utils import extract_last_class, report_results

from torch.nn.attention import sdpa_kernel, SDPBackend

from accelerate import Accelerator

import warnings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="VLM SFT training")

    parser.add_argument("--use_wandb", action="store_true", help="log to Weights & Biases")

    parser.add_argument("--device", type=str, default="cuda:0", help="torch device")
    parser.add_argument("--extra_line", type=str, default="tmp", help="suffix for save path")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--part_to_train", type=str, default="LoRA")
    parser.add_argument("--lr_schedule", type=str, default="linear", choices=["linear","cosine","constant"])
    parser.add_argument("--dataset_name", type=str, default="linesdetailed",
                        choices=[ "linesdetailed"])
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--user_system_message", type=str, default=None)
    parser.add_argument("--user_query_message", type=str, default=None)

    parser.add_argument("--test_size", type=int, default=5000)
    parser.add_argument("--train_size", type=int, default=656)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=55)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--processor_min_pixels", type=int, default=224*224)
    parser.add_argument("--processor_max_pixels", type=int, default=2048*2048)
    parser.add_argument("--evaluate_every_epoch", type=int, default=3)
    parser.add_argument("--test_subset_size", type=int, default=5000)

    parser.add_argument("--lora_dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-4)

    return parser.parse_args()


def zero_shot_predict(
        model, processor, test_dataset,
        query_idx,
        device="cuda",
        print_text=False,
        temperature=0.0,
        reps=1,
        system_message="",
        query_text="",
        max_new_tokens=512,
        all_labels=["FR-I", "FR-II"],
        model_type=None

):
    # assert system_message != "" and query_text != ""
    imgq = test_dataset[query_idx]["image"]

    msgs = [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
    ]
    msgs.extend([
        {"role": "user", "content": [{"type": "text", "text": query_text}, {"type": "image", "image": imgq}]},
    ])

    text_input = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(msgs)

    if print_text:
        print("Prompt:")
        print(text_input)

    inputs = processor(text=[text_input], images=image_inputs, return_tensors="pt").to(model.device)

    model = model.to(model.device)
    answers = Counter()
    raw_answers = []
    out_ids = model.generate(**inputs,
                             max_new_tokens=max_new_tokens,
                             pad_token_id=processor.tokenizer.pad_token_id,
                             num_return_sequences=reps,
                             )[:, inputs.input_ids.shape[-1]:]
    raw_decoded = processor.batch_decode(out_ids, skip_special_tokens=True)
    for raw in raw_decoded:
        raw = raw.strip()
        if print_text:
            print("Answer:")
            print(raw)
        raw_answers.append(raw)
        answer = extract_last_class(raw, all_labels=all_labels)
        answers[answer] += 1
    if print_text:
        print(answers)

    final_answer = answers.most_common(1)[0][0]

    res = {"final_answer": final_answer, "answers": answers, "raw_answers": raw_answers}

    return res

@torch.inference_mode()
def batched_zero_shot_predict(
    model, processor, dataset, idxs,
    system_message="", query_text="",
    batch_size=8, max_new_tokens=64, all_labels=("FR-I","FR-II"),
):
    device = getattr(model, "device", next(model.parameters()).device)
    preds = []

    for start in tqdm(range(0, len(idxs), batch_size)):
        chunk = idxs[start:start+batch_size]

        texts, imgs = [], []
        for i in chunk:
            img = dataset[i]["image"]  # PIL
            msgs = [
                {"role":"system","content":[{"type":"text","text":system_message}]},
                {"role":"user",  "content":[{"type":"text","text":query_text},
                                            {"type":"image","image":img}]},
            ]
            text_input = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(msgs)
            texts.append(text_input)
            imgs.append(image_inputs)      # list-of-lists is ok (1 image per sample)

        inputs = processor(text=texts, images=imgs, return_tensors="pt", padding=True)
        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        # length of each prompt to strip it off later
        pad_id = processor.tokenizer.pad_token_id
        prompt_lens = (inputs["input_ids"] != pad_id).sum(dim=1)

        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=pad_id,
                return_dict_in_generate=False,
            )

        gens = []
        for r, L in enumerate(prompt_lens.tolist()):
            gens.append(out[r, L:])
        decoded = processor.batch_decode(gens, skip_special_tokens=True)

        for s in decoded:
            s = s.strip()
            lab = extract_last_class(s, all_labels=all_labels)  # your helper
            preds.append(lab if lab is not None else "")

    return preds

def evaluate_score(
        model, processor,
        test_dataset,
        system_message, query_text,
        all_labels, device
):
    test_indexes = range(len(test_dataset))
    y_true_majority = []
    y_pred_majority = []

    for i in tqdm(test_indexes, desc="Gathering preds"):
        gt = test_dataset[i]["label"]
        ans = zero_shot_predict(
            model, processor,
            test_dataset,
            i,
            device=device, print_text=False, reps=1, temperature=0,
            max_new_tokens=200, system_message=system_message, query_text=query_text,
            all_labels=all_labels,
        )
        if i == 0:
            log.info(ans)
        pred = ans["final_answer"]
        if pred is None:
            pred = ""
        if pred.lower().startswith("answer"):
            pred = pred.split(":", 1)[1].strip()
        y_true_majority.append(gt)
        y_pred_majority.append(pred)

    score = report_results(y_true_majority, y_pred_majority, True, all_labels=all_labels)
    return score

def evaluate_score_batched(
        model, processor,
        test_dataset,
        system_message, query_text,
        all_labels, device
):

    idxs = list(range(len(test_dataset)))
    preds = batched_zero_shot_predict(
        model, processor, test_dataset, idxs,
        system_message=system_message, query_text=query_text,
        batch_size=32, max_new_tokens=4, all_labels=all_labels,
    )
    y_true = [test_dataset[i]["label"] for i in idxs]
    score = report_results(y_true, preds, True, all_labels=all_labels)
    return score

def main():
    args = parse_args()

    device = args.device
    use_wandb = args.use_wandb
    extra_line = args.extra_line
    model_id = args.model_id
    test_size = args.test_size
    train_size = args.train_size
    epochs = args.epochs
    part_to_train = args.part_to_train
    batch_size = args.batch_size
    lora_r = args.lora_r
    lora_alpha = args.lora_alpha
    lora_dropout = args.lora_dropout
    lr = args.lr
    lr_schedule = args.lr_schedule
    dataset_name = args.dataset_name
    seed = args.seed
    num_workers = args.num_workers
    min_pixels = args.processor_min_pixels
    max_pixels = args.processor_max_pixels
    evaluate_every_epoch = args.evaluate_every_epoch
    test_subset_size = args.test_subset_size
    set_seed(seed)

    lora_layers = [
                "attn.qkv",
                "attn.proj",
                "q_proj",
                "v_proj",
                "mlp.fc1",
                "mlp.fc2",
            ]
    system_message, query_text = "", ""
    if args.user_system_message is not None:
        system_message=args.user_system_message
    if args.user_query_message is not None:
        query_text=args.user_query_message
    train_system_message = system_message
    train_query_text = query_text

    if use_wandb:
        import wandb
        wandb.init(
            project=dataset_name+"-sft-VLM",
            config={
                "device": device,
                "extra_line": extra_line,
                "model_id": model_id,
                "test_size": test_size,
                "train_size": train_size,
                "epochs": epochs,
                "part_to_train": part_to_train,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "lora_layers": lora_layers,
                "batch_size": batch_size,
                "test_subset_size": test_subset_size,
                "lr": lr,
                "system_message": system_message,
                "query_text": query_text,
                "train_system_message": train_system_message,
                "train_query_text": train_query_text,
                "lr_schedule": lr_schedule,
                "dataset_name" : dataset_name,
                "seed":seed
            }
        )

    assert model_id in [
        "Qwen/Qwen2-VL-2B-Instruct",
        "Qwen/Qwen2-VL-7B-Instruct",
    ]

    accelerator = Accelerator(mixed_precision="bf16")
    model, processor = load_model(model_id, min_pixels=min_pixels, max_pixels=max_pixels)

    if dataset_name == "linesdetailed":
        datasets = load_lines_dataset_all()
        all_labels=["0", "1", "2"]

        label2id = {lbl: i for i, lbl in enumerate(all_labels)}

        def add_numeric_labels(example, label_key="label", new_key="label_nb"):
            example[new_key] = label2id[example[label_key]]
            return example

        for subset_name in datasets:
            datasets[subset_name] = datasets[subset_name].map(add_numeric_labels)

        test_dataset = datasets["paper_version"]

        for subset_name in datasets:
            if subset_name == "train":
                continue
            if subset_name != "paper_version":
                subset_size = test_subset_size
            else:
                subset_size = test_size
            print(subset_size, subset_name)

            n_totals = len(datasets[subset_name])
            print(n_totals)
            generator = torch.Generator().manual_seed(42)
            if subset_size < n_totals:
                log.info(f"Choosing {subset_size} from subset {subset_name}")
                datasets[subset_name], _ = random_split(
                    datasets[subset_name],
                    [subset_size, n_totals - subset_size],
                    generator=generator
                )


    train_dataset = datasets["train"]
    del datasets["train"]

    n_totals = len(train_dataset)
    generator = torch.Generator().manual_seed(42)

    train_subset, _ = random_split(
            train_dataset,
            [train_size, n_totals-train_size],
            generator=generator
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"{trainable_params}/{total_params}")

    losses = []

    if args.ckpt_dir is not None:

        ckpt_dir = args.ckpt_dir

        # Load the LoRA weights onto the base model
        model = PeftModel.from_pretrained(model, ckpt_dir, device_map="auto")

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info(f"Loaded LoRA from {ckpt_dir}  ->  {trainable_params}/{total_params}")
    else:
        model = configure_trainable_params(
            model, part_to_train,
            lora_r, lora_alpha, lora_dropout,
            lora_layers,
        )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"{trainable_params}/{total_params}")

    collate = Collator(pad_id=processor.tokenizer.pad_token_id)
    train_loader = DataLoader(
        QwenChatDataset(train_subset, processor, train_system_message, train_query_text),
        batch_size=batch_size, shuffle=True, collate_fn=collate, pin_memory=True, num_workers=num_workers,
    )

    num_update_steps_per_epoch = len(train_loader)
    max_train_steps = num_update_steps_per_epoch * (epochs + 1)
    num_warmup_steps = int(0.1 * max_train_steps)

    scheduler_type = lr_schedule
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    if scheduler_type == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=max_train_steps,
        )
    elif scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=max_train_steps,
        )
    elif scheduler_type == "constant":
        scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps
        )
    else:
        raise ValueError(f"Unknown scheduler_type: {scheduler_type}")

    RUN_ID = uuid.uuid4().hex[:8]
    CKPT_DIR = Path(f"./ckpt_{RUN_ID}")
    MILESTONE_EVERY = max(1, epochs//evaluate_every_epoch)
    EPS = 1e-8

    best_train_loss = float("inf")
    best_epoch = None
    best_test_metrics = None
    improved_since_last_milestone = False

    losses = []

    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    try:
        global_step = 0
        for epoch in range(0, epochs + 1):
            model.train()
            total_train = 0.0

            for batch in tqdm(train_loader, desc=f"Train {epoch}/{epochs}"):
                optimizer.zero_grad()

                out = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    pixel_values=batch['pixel_values'],
                    image_grid_thw=batch['image_grid_thw'],
                    labels=batch['labels']
                )

                loss = out.loss
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                total_train += float(loss.item())
                global_step += 1

            train_average_loss = total_train / max(1, len(train_loader))
            if use_wandb:
                wandb.log({"epoch": epoch, "train_average_loss": train_average_loss})
            log.info(f"Epoch {epoch} Train Loss: {train_average_loss:.6f}")

            if train_average_loss + EPS < best_train_loss:
                best_train_loss = train_average_loss
                best_epoch = epoch
                improved_since_last_milestone = True

                if CKPT_DIR.exists():
                    shutil.rmtree(CKPT_DIR, ignore_errors=True)
                CKPT_DIR.mkdir(parents=True, exist_ok=True)
                try:
                    model.save_pretrained(str(CKPT_DIR))
                    log.info(f"[epoch {epoch}] Saved new-best checkpoint -> {CKPT_DIR}")
                except Exception as e:
                    log.warning(f"Could not save checkpoint: {e}")
            else:
                print(train_average_loss, best_train_loss)

            if (epoch + 1) % MILESTONE_EVERY == 0:
                if improved_since_last_milestone:
                    try:
                        eval_model = None

                        eval_model, _ = load_model(model_id, device_map={"": model.device})
                        eval_model = PeftModel.from_pretrained(eval_model, str(CKPT_DIR))

                        eval_model.eval()

                        for dataset_name in datasets:
                            with torch.no_grad():
                                best_test_metrics = evaluate_score(
                                    eval_model, processor,
                                    datasets[dataset_name], system_message, query_text, all_labels, model.device
                                )

                            if use_wandb and isinstance(best_test_metrics, dict):
                                for k, v in best_test_metrics.items():
                                    wandb.log({"epoch": epoch, f"test_{k}({dataset_name})": v})

                        log.info(f"[Test metrics @ epoch {epoch} | reloaded best ckpt] {best_test_metrics}")

                    except Exception as e:
                        log.exception(f"Failed to reload/evaluate best checkpoint from {CKPT_DIR}: {e}")
                    finally:
                        if 'eval_model' in locals() and eval_model is not None:
                            del eval_model
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                    improved_since_last_milestone = False
                else:
                    log.info(f"[epoch {epoch}] No new best since last milestone — skipping TEST eval.")

            f1_test = (best_test_metrics or {}).get("f1", float('nan'))
            losses.append([train_average_loss, f1_test])

    except KeyboardInterrupt:
        losses = np.array(losses)
        log.info(losses)

    log.info(f"Best train loss {best_train_loss:.6f} at epoch {best_epoch}")

    if CKPT_DIR.exists():
        eval_model, _ = load_model(model_id, device_map={"": model.device})
        eval_model = PeftModel.from_pretrained(eval_model, str(CKPT_DIR))

        eval_model.eval()

        with torch.no_grad():
            best_test_metrics = evaluate_score(
                eval_model, processor,
                    test_dataset, system_message, query_text, all_labels, model.device
                )

            if use_wandb and isinstance(best_test_metrics, dict):
                for k, v in best_test_metrics.items():
                    wandb.log({"epoch": epoch, f"test_{k}(final score on the paper subset)": v})

        shutil.rmtree(CKPT_DIR, ignore_errors=True)
        log.info("Deleted checkpoint directory to keep the run clean.")

    losses = np.array(losses)
    log.info(losses)
    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()

