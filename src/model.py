
import warnings
import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen2VLProcessor, Qwen2VLForConditionalGeneration


def load_model(model_id, min_pixels=224 * 224, max_pixels=2048 * 2048, device_map="auto"):
    try:
        import flash_attn
        attn_impl = "flash_attention_2"
    except ImportError:
        warnings.warn(
            "FlashAttention 2 not installed — falling back to standard attention. "
            "Install for 2–4× speed-up: pip install flash-attn --no-build-isolation",
            UserWarning,
        )
        attn_impl = "eager"

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map=device_map,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation=attn_impl,
    )

    processor = Qwen2VLProcessor.from_pretrained(model_id, min_pixels=min_pixels, max_pixels=max_pixels)
    processor.tokenizer.padding_side = "left"
    return model, processor

def _is_lora_param(n: str) -> bool:
    return ("lora_A" in n) or ("lora_B" in n) or ("lora_embedding_A" in n) or ("lora_embedding_B" in n)

def configure_trainable_params(model, part_to_train, lora_r, lora_alpha, lora_dropout, lora_layers):
    VISION_KEYS = (".visual.", ".vision_tower.", ".vision_model.", ".vision.")
    def lora_wrap(m):
        return get_peft_model(m, LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=lora_layers,
        ))
    if part_to_train == "LoRA":
        return lora_wrap(model)
    elif part_to_train == "lora_to_decoder":
        model = lora_wrap(model)
        for p in model.parameters():
            p.requires_grad = False

    elif part_to_train == "lora_to_vision":
        model = lora_wrap(model)
        for p in model.parameters():
            p.requires_grad = False

        for n, p in model.named_parameters():
            if _is_lora_param(n) and any(k in n for k in VISION_KEYS):
                p.requires_grad = True

    elif part_to_train == "lora_to_decoder":
        model = lora_wrap(model)
        for p in model.parameters():
            p.requires_grad = False

        for n, p in model.named_parameters():
            in_vision = any(k in n for k in VISION_KEYS)
            if _is_lora_param(n) and not in_vision:
                p.requires_grad = True

    raise ValueError(f"Unknown part_to_train: {part_to_train}")