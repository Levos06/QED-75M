import argparse
import gc
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from sllm.model import SLLMForCausalLM
from sllm.config import ModelConfig as SLLMConfig

from lighteval.models.abstract_model import LightevalModel
from lighteval.models.transformers.transformers_model import TransformersModelConfig
from lighteval.tasks.registry import Registry
from lighteval.pipeline import Pipeline
from lighteval.models.model_output import ModelResponse
from lighteval.utils.cache_management import SampleCache
from lighteval.tasks.prompt_manager import PromptManager
from lighteval.metrics.metrics import Metrics

class SLLMLightevalWrapper(LightevalModel):
    def __init__(self, model_path: str, tokenizer_name: str, batch_size: int = 1, device: str = "cpu"):
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        sllm_config = SLLMConfig.from_dict(checkpoint["model_config"])
        self.model = SLLMForCausalLM(sllm_config)
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(device)
        self.model.eval()
        
        self._device = torch.device(device)
        self._sllm_config = sllm_config
        self.config = TransformersModelConfig(model_name="local-sllm")
        self._cache = SampleCache(self.config)
        self.batch_size = batch_size

        # Tokenizer setup
        import os
        if os.path.exists(tokenizer_name):
            # Load local BPE tokenizer
            from transformers import PreTrainedTokenizerFast
            self._tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_name)
            # Map special tokens from config
            self._tokenizer.bos_token = "<bos>"
            self._tokenizer.bos_token_id = sllm_config.bos_token_id
            self._tokenizer.eos_token = "<eos>"
            self._tokenizer.eos_token_id = sllm_config.eos_token_id
            self._tokenizer.pad_token = "<pad>"
            self._tokenizer.pad_token_id = sllm_config.pad_token_id
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Set padding side to left for generation
        self._tokenizer.padding_side = "left"

        self.prompt_manager = PromptManager(use_chat_template=False, tokenizer=self._tokenizer)

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def device(self):
        return self._device

    @property
    def max_length(self):
        return self._sllm_config.max_seq_len

    @property
    def add_special_tokens(self):
        return True

    def _apply_template(self, prompt: str) -> str:
        # Raw prompt for fair comparison with HF base models (same methodology)
        return prompt.strip()

    def greedy_until(self, requests):
        all_results = []
        # Process in batches
        for i in tqdm(range(0, len(requests), self.batch_size), desc="Generating", disable=self.disable_tqdm):
            batch_requests = requests[i : i + self.batch_size]
            batch_contexts = [self._apply_template(self.prompt_manager.prepare_prompt(r)) for r in batch_requests]
            
            # Use max(req.generation_size) for the whole batch
            max_new_tokens = max([(r.generation_size or 128) for r in batch_requests])
            stop_sequences = batch_requests[0].stop_sequences # Assume same for batch
            
            inputs = self.tokenizer(batch_contexts, return_tensors="pt", padding=True).to(self.device)
            
            generated = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=0,
                top_k=None,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            # Post-process batch
            for j, request in enumerate(batch_requests):
                input_len = inputs.input_ids.shape[1]
                response_ids = generated[j, input_len:]
                response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                
                # Truncate at stop sequences
                until = request.stop_sequences
                if until:
                    for stop in until:
                        if stop in response_text:
                            response_text = response_text.split(stop)[0]
                
                all_results.append(ModelResponse(text=[response_text]))
                
        return all_results

    def loglikelihood(self, requests):
        all_doc_logprobs = []
        # Flatten all (context, choice) pairs
        flattened_requests = []
        for request in requests:
            raw_context = self.prompt_manager.prepare_prompt(request)
            context = self._apply_template(raw_context)
            for choice in request.choices:
                flattened_requests.append((context, choice))

        # Process flattened requests in batches
        all_logprobs = []
        self.tokenizer.padding_side = "right"
        num_batches = (len(flattened_requests) + self.batch_size - 1) // self.batch_size
        for batch_idx, i in enumerate(tqdm(range(0, len(flattened_requests), self.batch_size), desc="LogLikelihood", total=num_batches, disable=self.disable_tqdm)):
            batch = flattened_requests[i : i + self.batch_size]
            
            batch_contexts = [b[0] for b in batch]
            batch_full_texts = [b[0] + b[1] for b in batch]
            
            ctx_enc = self.tokenizer(batch_contexts, return_tensors="pt", padding=True, add_special_tokens=self.add_special_tokens).to(self.device)
            full_enc = self.tokenizer(batch_full_texts, return_tensors="pt", padding=True, add_special_tokens=self.add_special_tokens).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(full_enc.input_ids, attention_mask=full_enc.attention_mask)
                # Extract logits and immediately free model outputs
                logits = outputs["logits"]
                del outputs
                
                # Shift for next-token prediction: predict token[i+1] from logits[i]
                shift_logits = logits[:, :-1, :].contiguous()
                del logits  # Free [B, S, V] immediately
                shift_labels = full_enc.input_ids[:, 1:].contiguous()
                
                # F.cross_entropy computes -log_softmax internally WITHOUT
                # materializing a full [B*S, V] softmax tensor in Python.
                # This halves peak memory vs explicit F.log_softmax + gather.
                flat_logits = shift_logits.view(-1, shift_logits.size(-1))
                seq_len = shift_logits.size(1)
                del shift_logits  # Free the shifted logits
                flat_labels = shift_labels.view(-1)
                
                per_token_nll = F.cross_entropy(flat_logits, flat_labels, reduction='none')
                del flat_logits, flat_labels  # Free immediately
                
                # Reshape back to [B, S-1] and convert NLL -> log-prob
                target_logprobs = -per_token_nll.view(len(batch), seq_len)
                del per_token_nll
                
                # Extract logprobs for the choice part only
                for j in range(len(batch)):
                    ctx_len = ctx_enc.attention_mask[j].sum().item()
                    full_len = full_enc.attention_mask[j].sum().item()
                    choice_lp = float(target_logprobs[j, ctx_len-1 : full_len-1].sum().item())
                    all_logprobs.append(choice_lp)
                
                del target_logprobs, shift_labels
            
            del ctx_enc, full_enc
            # Aggressively free MPS memory every 10 batches to prevent 16GB+ RAM growth
            if self._device.type == "mps" and batch_idx % 10 == 0:
                torch.mps.synchronize()
                torch.mps.empty_cache()
                gc.collect()

        # Unflatten
        idx = 0
        for request in requests:
            doc_probs = []
            for _ in request.choices:
                doc_probs.append(all_logprobs[idx])
                idx += 1
            all_doc_logprobs.append(ModelResponse(logprobs=doc_probs))
            
        return all_doc_logprobs

    def loglikelihood_rolling(self, requests):
        results = []
        for request in tqdm(requests, desc="LogLikelihood Rolling", disable=self.disable_tqdm):
            raw_text = self.prompt_manager.prepare_prompt(request)
            text = self._apply_template(raw_text)
            ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=self.add_special_tokens).input_ids.to(self.device)
            
            if ids.shape[1] <= 1:
                results.append(ModelResponse(logprobs=[0.0]))
                continue

            with torch.no_grad():
                outputs = self.model(ids)
                logits = outputs["logits"]
                
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = ids[..., 1:].contiguous()
            
            logprobs = F.log_softmax(shift_logits, dim=-1)
            target_logprobs = torch.gather(logprobs, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
            
            results.append(ModelResponse(logprobs=[float(target_logprobs.sum())]))
            
        return results


class HuggingFaceLightevalWrapper(SLLMLightevalWrapper):
    """Wrapper for HuggingFace CausalLM models (e.g. Pythia, GPT-2). Uses raw prompt, no chat template."""

    def __init__(self, model_name: str, batch_size: int = 1, device: str = "cpu"):
        self._device = torch.device(device)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self._device)
        self.model.eval()
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._tokenizer.padding_side = "left"
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        max_length = getattr(self.model.config, "max_position_embeddings", 2048)
        self._max_length = min(max_length, 2048)  # Cap for memory
        self.config = TransformersModelConfig(model_name=model_name.replace("/", "-"))
        self._cache = SampleCache(self.config)
        self.batch_size = batch_size
        self.prompt_manager = PromptManager(use_chat_template=False, tokenizer=self._tokenizer)

    @property
    def max_length(self):
        return self._max_length

    def _apply_template(self, prompt: str) -> str:
        # Base models (Pythia, GPT-2): use raw prompt, no chat formatting
        return prompt.strip()


def patch_tasks_to_loglikelihood():
    """Patch multi-choice tasks to use log-likelihood accuracy (classification)
    instead of generative exact-match. This is ~20x faster and gives real scores.
    Tasks that already have choices + gold_index in their Doc are compatible."""
    registry = Registry()
    acc_metric = Metrics.loglikelihood_acc.value  # SampleLevelMetric object
    patched = 0
    for name, task_config in registry._task_registry.items():
        # Patch multi-choice tasks: MMLU, HellaSwag, ARC, Winogrande, TruthfulQA:mc
        if (
            name.startswith("mmlu:")
            or name == "hellaswag"
            or name in ("arc:challenge", "arc:easy", "winogrande", "truthfulqa:mc")
        ):
            task_config.metrics = (acc_metric,)  # Tuple of SampleLevelMetric
            task_config.generation_size = -1  # Disable generation
            patched += 1
    print(f"Patched {patched} tasks to use log-likelihood accuracy (classification mode)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None, help="Path to local .pt checkpoint (use with tokenizer_name)")
    parser.add_argument("--model_name", type=str, default=None, help="HuggingFace model id (e.g. EleutherAI/pythia-160m-deduped); overrides model_path")
    parser.add_argument("--tokenizer_name", type=str, default="/Users/levosadchi/Desktop/transformer4/data/tokenizer/tokenizer.json")
    parser.add_argument("--tasks", type=str, default="gsm8k")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--sequential_tasks", action="store_true", help="Run each task in separate pipeline to free memory between tasks (avoids 16GB+ RAM growth)")
    parser.add_argument("--classification_only", action="store_true", help="Skip generative tasks (gsm8k, truthfulqa:gen); only run loglikelihood/classification tasks")
    args = parser.parse_args()

    if args.device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # Patch MMLU/HellaSwag to use log-likelihood instead of generation
    patch_tasks_to_loglikelihood()

    from lighteval.pipeline import Pipeline, PipelineParameters, ParallelismManager
    from lighteval.logging.evaluation_tracker import EvaluationTracker

    # 1. Initialize Wrapper
    if args.model_name:
        if args.model_path:
            print("Warning: model_name overrides model_path")
        print(f"Loading HuggingFace model: {args.model_name}")
        wrapper = HuggingFaceLightevalWrapper(args.model_name, batch_size=args.batch_size, device=device)
    elif args.model_path:
        wrapper = SLLMLightevalWrapper(args.model_path, tokenizer_name=args.tokenizer_name, batch_size=args.batch_size, device=device)
    else:
        raise ValueError("Provide either --model_path or --model_name")

    # 2. Initialize Evaluation Tracker
    evaluation_tracker = EvaluationTracker(
        output_dir=args.output_dir,
        save_details=True,
    )


    # 3. Initialize Pipeline Parameters
    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.NONE,
        max_samples=args.max_samples,
    )

    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]

    if args.classification_only:
        GENERATIVE_TASKS = {"gsm8k", "truthfulqa:gen"}
        kept = [t for t in task_list if t not in GENERATIVE_TASKS]
        removed = [t for t in task_list if t in GENERATIVE_TASKS]
        if removed:
            print(f"Classification-only mode: skipping {removed}")
        task_list = kept
        args.tasks = ",".join(task_list)
        if not task_list:
            print("No tasks left after filtering. Exiting.")
            return

    if args.sequential_tasks and len(task_list) > 1:
        # Run each task in separate pipeline to free memory between tasks
        print(f"Sequential mode: running {len(task_list)} tasks one by one to reduce memory usage")
        for i, task in enumerate(task_list):
            print(f"\n--- Task {i+1}/{len(task_list)}: {task} ---")
            task_tracker = EvaluationTracker(output_dir=args.output_dir, save_details=True)
            pipeline = Pipeline(
                tasks=task,
                pipeline_parameters=pipeline_params,
                evaluation_tracker=task_tracker,
                model=wrapper,
            )
            pipeline.evaluate()
            pipeline.save_and_push_results()
            pipeline.show_results()
            del pipeline
            del task_tracker
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()
    else:
        # Single pipeline for all tasks
        pipeline = Pipeline(
            tasks=args.tasks,
            pipeline_parameters=pipeline_params,
            evaluation_tracker=evaluation_tracker,
            model=wrapper,
        )
        pipeline.evaluate()
        pipeline.save_and_push_results()
        pipeline.show_results()

if __name__ == "__main__":
    main()
