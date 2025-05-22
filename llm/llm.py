# pip install transformers==4.47.1
# pip install transformers==4.46.1
import numpy as np
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    logging,
    TrainerCallback
)
import evaluate
import torch
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import wandb
import argparse
from tqdm import tqdm
from copy import deepcopy
from huggingface_hub import login
import gc
import os
from sklearn.metrics import accuracy_score
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.nist_score import sentence_nist
from dataclasses import dataclass, field
from transformers import HfArgumentParser
import socket
from peft import prepare_model_for_kbit_training
from evaluate import load as load_metric

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # use single GPU


# Constants and configurations
DEBUG_MODE = True  # Set to False for full dataset
IFNOTQUNT = False
DEBUG_SIZE = 500  # Limit dataset size for debuging
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"

os.environ["TOKENIZERS_PARALLELISM"] = "false" # Avoid tokenizer deadlocks after fork
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" # reduce memory fragmentation

# Model and dataset configuration
MODEL_DATA_CONFIG = {
    "model": {
        "meta-llama/Llama-2-7b-hf": "meta-llama/Llama-2-7b-hf",
        "meta-llama/Meta-Llama-3-8B": None,
        "Llama-3.3-70B-Instruct": None,
        "Llama-3.2-1B": "./.llama/checkpoints/Llama-3.2-1B", # update model address
        "Llama-3.2-1B-anl": "/models/checkpoints/Llama-3.2-1B", # update model address
        "gpt2": "gpt2",
        "gpt2-medium": "gpt2-medium",
        "gpt2-large": "gpt2-large",
        "Llama-3.2-3B": "meta-llama/Llama-3.2-3B",
        "Llama-3.1-70B": "meta-llama/Llama-3.1-70B",
        "Llama-3.1-8B": "meta-llama/Llama-3.1-8B",
        "Llama-3.3-70B-Instruct": "meta-llama/Llama-3.3-70B-Instruct"
    },
    "dataset": {
        "mlabonne/guanaco-llama2-1k": None,
        "alpaca": "tatsu-lab/alpaca",
        "e2e_nlg": None,
    }
}


e2e_dataset = load_dataset("e2e_nlg", split="test")

@dataclass
class ModelArguments:
    eval_solar: bool = field(default=True)
    solar_retain_params: float = field(default=0.3)
    solar_random_basis: int = field(default=100)
    solar_basis_per_vector: int = field(default=20)
    solar_scaling: float = field(default=1)
    solar_use_similarity: bool = field(default=False)
    num_train_epochs: int = field(default=5) # num epochs

    learning_rate: float = field(default=0.1) # llama :2e-5

    lora_r: int = field(
        default=4,
        metadata={"help": "Lora R dimension."}
    )
    lora_alpha: float = field(
        default=8,
        metadata={"help": "Lora alpha."}
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help": "Lora dropout."}
    )


@dataclass
class CustomArgs:
    mmlu_dataset: str = field(default="mmlu-fs")
    mmlu_split: str = field(default="eval")
    # Llama-3.2-3B, Llama-3.2-1B, Llama-3.1-70B, Llama-3.1-8B, gpt2, "gpt2-medium", "gpt2-large"
    model_key: str = field(default="gpt2")
    do_mmlu_eval: bool = field(default=False)
    max_mmlu_samples: int = field(default=DEBUG_SIZE if DEBUG_MODE else None)
    mmlu_source_max_len: int = field(default=256) #  change from orignal 2024
    per_device_train_batch_size: int = field(default=8)
    gradient_accumulation_steps: int = field(default=16)
    ifQuantization: bool = field(default=False)

class AccuracyCallback(TrainerCallback):
    def __init__(self, train_dataset, eval_dataset, tokenizer, model):
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.model = model

    def on_epoch_end(self, args, state, control, **kwargs):
        return


class CustomCallback(TrainerCallback):
    def __init__(self, *args, model, retain_params=10, trainer=None, mmlu_dataset=None,
                 tokenizer=None, model_args=None, custom_args=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.retain_params = retain_params
        self.model_original = model
        self.trainer = trainer
        self.mmlu_dataset = mmlu_dataset
        self.tokenizer = tokenizer
        self.model_args = model_args
        self.custom_args = custom_args


    def on_epoch_end(self, args, state, control, **kwargs):
        """Runs evaluation, pruning, and masking at the end of each epoch."""
        # trainer = kwargs['trainer']
        trainer = self.trainer
        device = trainer.args.device
        print(f"Epoch {state.epoch} ended. Running evaluation ...")

        print(f"Trainer model device: {next(trainer.model.parameters()).device}")

        metrics_original = trainer.evaluate()

        # Run SOLAR evaluation
        solar_model = SOLAR(
            trainer.model,
            num_random_basis=self.model_args.solar_random_basis,
            num_random_basis_each_vector=self.model_args.solar_basis_per_vector,
            random_bases_scaling=self.model_args.solar_scaling,
            use_similar_vectors=self.model_args.solar_use_similarity,
            model_path=self.custom_args.model_path,
        ).to(trainer.args.device)

        print(f"SOLAR model device: {next(solar_model.parameters()).device}")

        original_model = trainer.model
        trainer.model = solar_model


        metrics_masking = trainer.evaluate(metric_key_prefix="SOLAR")

        # Cleanup
        del solar_model
        gc.collect()
        torch.cuda.empty_cache()

        # === MMLU Accuracy Evaluation at Epoch End ===
        if self.custom_args.do_mmlu_eval:
            try:
                mmlu_dataset = load_dataset("json", data_files={
                    'eval': 'data/mmlu/five_shot_mmlu_val.json',
                    'test': 'data/mmlu/five_shot_mmlu_test.json',
                })
                mmlu_dataset = mmlu_dataset[self.custom_args.mmlu_split]

                if self.custom_args.max_mmlu_samples:
                    mmlu_dataset = mmlu_dataset.select(range(self.custom_args.max_mmlu_samples))
                inputs = [ex["input"] for ex in mmlu_dataset]
                targets = [ord(ex["output"]) - ord("A") for ex in mmlu_dataset]

                abcd = ["A", "B", "C", "D"]
                tokenizer = self.tokenizer
                abcd_ids = [tokenizer(a, add_special_tokens=False).input_ids[0] for a in abcd]

                def eval_mmlu(model, inputs, targets, abcd_ids, tokenizer, batch_size=1):
                    model.eval()
                    preds = []
                    losses = []
                    loss_fn = torch.nn.CrossEntropyLoss()

                    with torch.no_grad():
                        for i in range(0, len(inputs), batch_size):
                            batch_inputs = inputs[i:i + batch_size]
                            encoded = tokenizer(
                                batch_inputs,
                                return_tensors='pt',
                                padding=True,
                                truncation=True,
                                max_length=2048
                            ).to(model.device)

                            outputs = model(**encoded)
                            logits = outputs.logits
                            last_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]

                            batch_logits = last_token_logits[:, abcd_ids]  # [batch_size, 4]

                            batch_targets = torch.tensor(targets[i:i + batch_size], device=model.device)
                            loss = loss_fn(batch_logits, batch_targets)

                            batch_preds = torch.argmax(batch_logits, dim=1).cpu().tolist()
                            preds.extend(batch_preds)
                            losses.append(loss.item())

                    acc = accuracy_score(targets, preds)
                    avg_loss = np.mean(losses)

                    return acc, avg_loss

                acc_original, loss_original = eval_mmlu(self.model_original, inputs, targets, abcd_ids, tokenizer)
                acc_solar, loss_solar = eval_mmlu(self.trainer.model, inputs, targets, abcd_ids, tokenizer)

                print(f"[Epoch {state.epoch}] MMLU Accuracy (Original): {acc_original:.4f}")
                print(f"[Epoch {state.epoch}] MMLU Accuracy (SOLAR): {acc_solar:.4f}")

                if "wandb" in self.trainer.args.report_to:
                    diff_acc = acc_solar - acc_original

                    wandb.log({
                        "mmlu_accuracy/original": acc_original,
                        "mmlu_accuracy/solar": acc_solar,
                        "mmlu_accuracy/difference": diff_acc,
                        "mmlu_loss/original": loss_original,
                        "mmlu_loss/solar": loss_solar,
                        "solar/retain_params": self.model_args.solar_retain_params,
                        "solar/num_random_basis": self.model_args.solar_random_basis,
                        "epoch": state.epoch
                    })

            except Exception as e:
                print("MMLU Eval failed:", e)

        # Restore original model
        trainer.model = self.model_original

def re_quantize_model(model, model_name: str):
    """Re-quantize the model after modifications"""
    if IFNOTQUNT:
        return model
    model.save_pretrained("temp_model")
    re_quantized_model = AutoModelForCausalLM.from_pretrained(
        "temp_model",
        quantization_config=quant_config,
        device_map="auto" # ={"": 0},  for single GPU
    )
    return re_quantized_model

def SOLAR(
    model,
    retain_params=0.3,
    num_random_basis=1000,
    num_random_basis_each_vector=20,
    random_bases_scaling=1,
    use_similar_vectors=True,
    model_path=None,
):
    updated_model = deepcopy(model)
    device = next(model.parameters()).device

    if not IFNOTQUNT:
        torch.cuda.empty_cache()
        gc.collect()
        print("[SOLAR] Loading full-precision model for dequantized W ...")
        float_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            # torch_dtype=torch.float32,
            torch_dtype=torch.float16,
            # device_map={"": device}
            device_map="cpu" # as just want for svd not training
        )
        float_modules = dict(float_model.named_modules())


    def generate_basis(proj, base_vectors, is_A=True):
        N, out_dim, in_dim = num_random_basis, proj.shape[0], proj.shape[1]
        M = torch.zeros(N, out_dim, in_dim, device=device)
        num_selected_vectors = out_dim if is_A else in_dim

        proj = proj if is_A else proj.T

        for i in range(N):
            if use_similar_vectors:
                distances = torch.cdist(proj, base_vectors.T, p=2)
                indices = torch.topk(distances, k=min(num_selected_vectors, base_vectors.shape[1]),
                                     largest=False).indices

            else:
                indices = torch.randint(0, base_vectors.shape[1], (num_selected_vectors,), device=device)
            base_vecs = base_vectors[:, indices]
            noise = torch.randn_like(base_vecs, device=device)
            perturbed = base_vecs + random_bases_scaling * noise
            M[i] = perturbed.T if is_A else perturbed
        return M

    def optimize_proj(M, target, retain_count):
        device = M.device
        M_flat = M.view(M.shape[0], -1).T.to(device)
        target_flat = target.flatten().to(device)
        sol = torch.linalg.lstsq(M_flat, target_flat).solution
        top_k = torch.topk(torch.abs(sol), retain_count).indices
        mask = torch.zeros_like(sol, device=device)
        mask[top_k] = 1

        return (M_flat @ (sol * mask)).view_as(target_flat)

    for name, module in updated_model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B') and hasattr(module, 'weight'):
            A = module.lora_A.default.weight.to(torch.float32).to(device)
            B = module.lora_B.default.weight.to(torch.float32).to(device)

            if hasattr(module.weight, "dequantize") and not IFNOTQUNT:
                float_name = name.replace("base_model.model.model.", "model.")
                W = float_modules[float_name].weight.data.to(torch.float32).to(device)
            else:
                W = module.weight.to(torch.float32).to(device)

            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            Vt = Vt.to(device)
            U = U.to(device)
            V = Vt.T
            if "gpt2" in model_path.lower():
                A, B = B.T, A.T
            A_proj = A @ V # to handle non square W
            B_proj = U.T @ B

            retain_count = min(int(retain_params * num_random_basis), num_random_basis)

            M_A = generate_basis(A_proj, Vt, is_A=True) # to handle non square W
            M_B = generate_basis(B_proj, U, is_A=False)

            A_approx = optimize_proj(M_A, A_proj, retain_count).view_as(A_proj) @ Vt # to handle non square W
            B_approx = U @ optimize_proj(M_B, B_proj, retain_count).view_as(B_proj)

            if "gpt2" in model_path.lower():
                A_approx, B_approx = B_approx.T, A_approx.T
            module.lora_A.default.weight.data.copy_(A_approx)
            module.lora_B.default.weight.data.copy_(B_approx)

    updated_model = updated_model.to(device)
    return updated_model



def initialize_model_and_tokenizer(model_path, if_quant):
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if if_quant:
        print("Quantization is disabled.")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            # gradient_checkpointing=True,
            trust_remote_code=True,
        )
    else:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quant_config,
            device_map="auto",
            token=True,
            trust_remote_code=True,
            torch_dtype=None,
        )
        model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    model.config.use_cache = False
    model.config.pretraining_tp = 1
    return model, tokenizer


def setup_datasets(dataset_name, debug_mode=False, debug_size=2):
    dataset = load_dataset(dataset_name, split='train')
    train_dataset = dataset.select(range(int(len(dataset) * 0.8)))
    eval_dataset = dataset.select(range(int(len(dataset) * 0.8), len(dataset)))

    if debug_mode:
        train_dataset = train_dataset.select(range(min(debug_size, len(train_dataset))))
        eval_dataset = eval_dataset.select(range(min(debug_size, len(eval_dataset))))

    return train_dataset, eval_dataset


def setup_training(model, train_dataset, eval_dataset, tokenizer, model_path, dataset_name,
                   model_args, mmlu_dataset=None, custom_args=None):
    lora_r = model_args.lora_r
    lora_alpha = model_args.lora_alpha
    lora_dropout = model_args.lora_dropout

    if "gpt2" in model_path.lower():
        global IFNOTQUNT
        IFNOTQUNT = True  # Disable quantization for GPT2
        lora_target_modules = ["c_attn"] # ["c_attn", "c_proj", "c_fc"]
    else:
        lora_target_modules = None

    lora_config = LoraConfig(
        lora_alpha=lora_alpha,
        r=lora_r,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_target_modules
    )


    peft_model = get_peft_model(model, lora_config)

    print_trainable_parameters(peft_model)

    training_args = TrainingArguments(
        output_dir="./results",
        run_name="Fast-PEFT-Llama",
        eval_strategy="epoch",
        eval_steps=20,
        learning_rate=model_args.learning_rate, # 2e-5
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        # num_train_epochs=10,
        fp16=True,
        fp16_full_eval=False,
        save_total_limit=2,
        logging_steps=10,
        save_steps=1000,
        report_to="wandb",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
    )

    model_id = [k for k, v in MODEL_DATA_CONFIG["model"].items() if v == model_path][0]
    dataset_id = [k for k, v in MODEL_DATA_CONFIG["dataset"].items() if v == dataset_name][0]

    wandb.init(
        project="Fast-PEFT",
        name=f"{model_id} - {dataset_id}",
        config={
            "model_key": custom_args.model_key,
            "model_name": model_path,
            "dataset_name": dataset_name,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            # "learning_rate": training_args.learning_rate,
            "solar_random_basis": model_args.solar_random_basis,
            "solar_retain_params": model_args.solar_retain_params,
            "learning_rate": model_args.learning_rate,
        }
    )

    custom_callback = CustomCallback(
        model=peft_model,
        retain_params=model_args.solar_retain_params,
        mmlu_dataset=mmlu_dataset,
        tokenizer=tokenizer,
        model_args=model_args,
        custom_args=custom_args
    )

    sft_config = SFTConfig(
        output_dir="output_dir",
        per_device_train_batch_size=16,
        num_train_epochs=model_args.num_train_epochs,
        logging_steps=10,
        max_seq_length=custom_args.mmlu_source_max_len,
        dataset_text_field="text",
        learning_rate=model_args.learning_rate,
    )

    trainer = SFTTrainer(
        model=peft_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_config,
        callbacks=[custom_callback]
    )

    custom_callback.trainer = trainer

    return trainer


def print_trainable_parameters(model):
    trainable_params = 0
    all_params = 0
    lora_params = 0
    mlp_head_params = 0

    for name, param in model.named_parameters():
        numel = param.numel()
        all_params += numel
        if param.requires_grad:
            trainable_params += numel

            if "lora_" in name:
                lora_params += numel
            elif any(kw in name.lower() for kw in ["mlp", "ffn", "dense", "head", "classifier", "output"]):
                mlp_head_params += numel

    print(f"Total trainable params: {trainable_params} / {all_params} ({100 * trainable_params / all_params:.2f}%)")
    print(f"  └── LoRA trainable params: {lora_params}")
    print(f"  └── MLP/Head trainable params: {mlp_head_params}")


def main():
    torch.cuda.empty_cache()

    login(token='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', add_to_git_credential=False)

    parser = HfArgumentParser((ModelArguments, CustomArgs))
    model_args, args = parser.parse_args_into_dataclasses()

    mmlu_dataset = None

    if args.do_mmlu_eval:

        mmlu_dataset = load_dataset("json", data_files={
            'eval': 'data/mmlu/five_shot_mmlu_val.json',
            'test': 'data/mmlu/five_shot_mmlu_test.json',
        })
        mmlu_dataset = mmlu_dataset[args.mmlu_split]

    selected_model = args.model_key
    if "gpt2" in selected_model.lower():
        global IFNOTQUNT
        IFNOTQUNT = True

    if '053426c7c714' not in socket.gethostname(): # avaialbe GPU at my side
        selected_model = "Llama-3.2-1B-anl"

    selected_dataset = "alpaca"
    model_path = MODEL_DATA_CONFIG["model"][selected_model]

    print(f"Selected model path: {model_path}")


    dataset_name = MODEL_DATA_CONFIG["dataset"][selected_dataset]

    model, tokenizer = initialize_model_and_tokenizer(model_path, IFNOTQUNT)

    print_trainable_parameters(model)

    train_dataset, eval_dataset = setup_datasets(dataset_name, DEBUG_MODE, DEBUG_SIZE)

    retain_params = int(model_args.solar_retain_params)

    args.model_path = model_path

    # Setup training
    trainer = setup_training(
        model,
        train_dataset,
        eval_dataset,
        tokenizer,
        model_path,
        dataset_name,
        model_args,
        # retain_params,
        mmlu_dataset=mmlu_dataset,
        custom_args=args
    )

    # Train the model
    train_results = trainer.train()
    wandb.log({"train_loss": train_results.metrics['train_loss']})

    del train_results
    gc.collect()
    torch.cuda.empty_cache()

    # Evaluate the model
    eval_results = trainer.evaluate()
    wandb.log({"eval_loss": eval_results['eval_loss']})

    del eval_results
    gc.collect()
    torch.cuda.empty_cache()

    if "gpt2" in model_path.lower():
        subset_frac = 0.001 if DEBUG_MODE else 1.0
        e2e_dataset = load_dataset("e2e_nlg", split="test")
        subset_size = int(len(e2e_dataset) * subset_frac)
        e2e_subset = e2e_dataset.select(range(subset_size))
        inputs = [ex["meaning_representation"] for ex in e2e_subset]
        references = [[ex["human_reference"]] for ex in e2e_subset]  # List of lists for corpus-level metrics

        if model_args.eval_solar:
            print("Running final SOLAR evaluation after training...")
            solar_model = SOLAR(
                model=trainer.model,
                retain_params=model_args.solar_retain_params,
                num_random_basis=model_args.solar_random_basis,
                num_random_basis_each_vector=model_args.solar_basis_per_vector,
                random_bases_scaling=model_args.solar_scaling,
                use_similar_vectors=model_args.solar_use_similarity,
                model_path=args.model_path,
            ).to(trainer.args.device)

            trainer.model = solar_model
            trainer.model.eval()

            solar_eval_results = trainer.evaluate(metric_key_prefix="SOLAR_final")
            if "SOLAR_final_eval_loss" in solar_eval_results:
                wandb.log({"SOLAR_final/eval_loss": solar_eval_results["SOLAR_final_eval_loss"]})
            else:
                print("Warning: 'SOLAR_final_eval_loss' not found in evaluation results.")
                print(f"Available keys: {list(solar_eval_results.keys())}")
            print("Final SOLAR eval results:")
            wandb.log({f"SOLAR_final/{k}": v for k, v in solar_eval_results.items()})
            for k, v in solar_eval_results.items():
                print(f"{k}: {v}")

            predictions = []
            for inp in tqdm(inputs, desc=f"Generating E2E outputs on {subset_size} samples (SOLAR)"):
                encoded = tokenizer(inp, return_tensors="pt", padding=True, truncation=True).to(solar_model.device)
                with torch.no_grad():
                    output_ids = solar_model.generate(
                        input_ids=encoded["input_ids"],
                        attention_mask=encoded["attention_mask"],
                        max_new_tokens=60,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                predictions.append(pred)

            metrics = {
                "meteor": load_metric("meteor"),
                
            }
            results = {}
            for name, metric in metrics.items():
                result = metric.compute(predictions=predictions, references=references)
                results[name] = result
                wandb.log({f"SOLAR_final/E2E/{name}": result})
                print(f"{name.upper()} (SOLAR): {result}")

            print("SOLAR E2E NLG Metrics Summary:")
            meteor = results["meteor"]["meteor"]
            print(f"METEOR: {meteor * 100:.2f}%")
            wandb.log({
                "SOLAR_final/E2E/METEOR (%)": meteor * 100,
            })

            trainer.model = model


        print("Running original model E2E evaluation...")
        model.eval()
        predictions = []
        for inp in tqdm(inputs, desc=f"Generating E2E outputs on {subset_size} samples (original)"):
            encoded = tokenizer(inp, return_tensors="pt", padding=True, truncation=True).to(model.device)
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    max_new_tokens=60,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                )
            pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            predictions.append(pred)

        metrics = {
            "meteor": load_metric("meteor"),
        }
        results = {}
        for name, metric in metrics.items():
            result = metric.compute(predictions=predictions, references=references)
            results[name] = result
            wandb.log({f"original/E2E/{name}": result})
            print(f"{name.upper()} (original): {result}")

        print("Original Model E2E NLG Metrics Summary:")
        meteor = results["meteor"]["meteor"]
        print(f"METEOR: {meteor * 100:.2f}%")
        wandb.log({
            "original/E2E/METEOR (%)": meteor * 100,
        })


    wandb.finish()


if __name__ == "__main__":
    main()
