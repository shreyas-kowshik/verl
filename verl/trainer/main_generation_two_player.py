# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Generate responses using two-player setup: example generator + solution generator
"""

import os

import hydra
import numpy as np
import ray
import torch

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from pprint import pprint

import pandas as pd
from omegaconf import OmegaConf

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.hdfs_io import makedirs
from verl.utils.model import compute_position_id_with_mask
from verl.workers.fsdp_workers import ActorRolloutRefWorker
import verl.utils.torch_functional as verl_F
import ray.util.rpdb as ray_pdb

# Example generator prompt template (from ray_trainer.py)
EXAMPLE_GENERATOR_PROMPT_TEMPLATE = """Given the following problem statement:
{problem_statement}

Generate a clear and complete solved example for this problem.
	•	Do not write any code.
	•	Choose simple toy values to illustrate the process.
	•	Show the step-by-step reasoning used to solve the example.
	•	Clearly present the initial input, intermediate steps, and final output.
	•	Format the solution neatly using bullet points or equations where appropriate.

Structure your response with the following sections:
	1.	Problem Recap
	2.	Example Input
	3.	Step-by-Step Solution
	4.	Final Answer

Generated Solved Example:

"""

# Default solution prompt template (from ray_trainer.py)
DEFAULT_SOLUTION_PROMPT_TEMPLATE = """You are a helpful Python coding assistant.
Given a task, output ONLY valid Python code that defines the required function(s).
Do not include explanations or markdown fences.
Do not include any docstrings or anything other than the python function(s).
Enclose the entire solution within ```python and ```.

{problem}

Constraints:
- Write clean, minimal Python and no other language.
- Define the function(s) exactly as implied by the tests.
- Do NOT print; just return values.
- Output ONLY code (no backticks, no explanations).
- Enclose the entire solution within ```python and ```.



To aid you in solving the problem, here is(are) step by step solved examples to illustrate how to solve the problem




{example}

"""


def format_problem_for_example_generator(problem_texts_raw: list[str]) -> list[str]:
    """Format raw problem texts for the example generator.
    
    Args:
        problem_texts_raw: List of raw problem texts extracted from prompts
        
    Returns:
        List of formatted prompts for the example generator
    """
    formatted_prompts = []
    for problem_text_raw in problem_texts_raw:
        prompt = EXAMPLE_GENERATOR_PROMPT_TEMPLATE.format(problem_statement=problem_text_raw)
        formatted_prompts.append(prompt)
    return formatted_prompts


def format_conditioned_prompt(problem_text: str, example_text: str, solution_prompt_template: str = None) -> str:
    """Format the prompt for solution generator by combining problem and example.
    
    Args:
        problem_text: The problem text
        example_text: The generated example text
        solution_prompt_template: Optional custom template, uses default if None
        
    Returns:
        Formatted prompt for the solution generator
    """
    template = solution_prompt_template or DEFAULT_SOLUTION_PROMPT_TEMPLATE
    extracted_problem_text = problem_text.split("Task:\n")[1].split("\n")[0]
    processed_problem_text = "Given the following problem statement:\nTask:\n" + extracted_problem_text
    return template.format(problem=processed_problem_text, example=example_text)


def process_generated_text_to_tensors(
    generated_text: list[str],
    tokenizer,
    max_prompt_length: int,
    truncation: str = "right",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
    """Process generated text into tensors for model input.
    
    Args:
        generated_text: List of text prompts to process
        tokenizer: The tokenizer to use
        max_prompt_length: Maximum prompt length
        truncation: Truncation mode ("left", "right", "middle", or "error")
        
    Returns:
        Tuple of (input_ids, attention_mask, position_ids, raw_prompt_ids)
    """
    if hasattr(tokenizer, 'apply_chat_template'):
        all_messages = []
        for text in generated_text:
            messages = [{"role": "user", "content": text}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            all_messages.append(text)
    else:
        all_messages = generated_text
    
    model_inputs = tokenizer(
        all_messages, 
        return_tensors="pt", 
        add_special_tokens=False,
        padding="longest",
        truncation=False,
    )
    
    input_ids = model_inputs.pop("input_ids")
    attention_mask = model_inputs.pop("attention_mask")

    input_ids, attention_mask = verl_F.postprocess_data(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_prompt_length,
        pad_token_id=tokenizer.pad_token_id,
        left_pad=True,
        truncation=truncation,
    )

    position_ids = compute_position_id_with_mask(attention_mask)

    def _truncate_ids(raw_ids):
        if len(raw_ids) <= max_prompt_length:
            return raw_ids

        if truncation == "left":
            return raw_ids[-max_prompt_length:]
        elif truncation == "right":
            return raw_ids[:max_prompt_length]
        elif truncation == "middle":
            left_half = max_prompt_length // 2
            right_half = max_prompt_length - left_half
            return raw_ids[:left_half] + raw_ids[-right_half:]
        elif truncation == "error":
            raise RuntimeError(
                f"Prompt length {len(raw_ids)} is longer than {max_prompt_length}."
            )
        else:
            raise ValueError(f"Unknown truncation mode: {truncation!r}")

    all_prompt_ids = []
    for text in all_messages:
        raw_ids = tokenizer.encode(text, add_special_tokens=False)
        raw_ids = _truncate_ids(raw_ids)
        all_prompt_ids.append(raw_ids)
    
    return input_ids, attention_mask, position_ids, all_prompt_ids


@hydra.main(config_path="config", config_name="generation_two_player", version_base=None)
def main(config):
    run_generation_two_player(config)


def run_generation_two_player(config) -> None:
    """Main entry point for two-player generation."""
    if not ray.is_initialized():
        # this is for local ray cluster
        default_runtime_env = {"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}}
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    ray.get(main_task.remote(config))


@ray.remote(num_cpus=1)
def main_task(config):
    """Main task for two-player generation running on Ray."""
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    # Load tokenizers for both models
    example_model_path = copy_to_local(config.example_model.path)
    solution_model_path = copy_to_local(config.solution_model.path)
    
    trust_remote_code = config.data.get("trust_remote_code", False)
    example_tokenizer = hf_tokenizer(example_model_path, trust_remote_code=trust_remote_code)
    solution_tokenizer = hf_tokenizer(solution_model_path, trust_remote_code=trust_remote_code)

    # Set padding side and pad token
    example_tokenizer.padding_side = "left"
    if example_tokenizer.pad_token is None:
        example_tokenizer.pad_token = example_tokenizer.eos_token
    
    solution_tokenizer.padding_side = "left"
    if solution_tokenizer.pad_token is None:
        solution_tokenizer.pad_token = solution_tokenizer.eos_token

    # Get sample counts - support both old detailed format and new simplified format
    n_example_samples = config.data.get("n_example_samples", 1)
    n_solution_samples = config.data.get("n_solution_samples", config.data.get("n_samples", 5))
    
    # Validate config
    if config.example_rollout.temperature == 0.0:
        assert n_example_samples == 1, "When temperature=0 for example generator, n_example_samples must be 1."
    if config.solution_rollout.temperature == 0.0:
        assert n_solution_samples == 1, "When temperature=0 for solution generator, n_solution_samples must be 1."

    # Read dataset
    dataset = pd.read_parquet(config.data.path)
    
    # Extract problem texts
    prompt_key = config.data.get("prompt_key", config.data.get("problem_key", "prompt"))
    problem_texts_raw = dataset[prompt_key].tolist()

    # Process problem texts to extract just the task portion
    # Assuming format like: "Task: <task_text> Constraints: <constraints>"
    problem_texts_extracted = []
    for text in problem_texts_raw:
        text = text[0]['content']
        if "Task:" in text:
            extracted = "Task:" + text.split("Task:", 1)[1].split("Constraints:", 1)[0]
        else:
            extracted = text
        problem_texts_extracted.append(extracted)

    # Initialize Example Generator Worker Group
    print("Initializing Example Generator Worker Group...")
    # Build example config by copying the base config structure
    # This ensures all required fields are present
    example_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    example_config.model = config.example_model
    example_config.rollout = config.example_rollout
    example_config.actor = config.example_actor
    # example_config.ref = config.example_rollout.ref
    
    # Get CPU count per GPU (default to 4 if not specified)
    example_n_cpus_per_gpu = config.trainer.get("example_n_cpus_per_gpu", 4)
    
    example_ray_cls = RayClassWithInitArgs(
        cls=ray.remote(ActorRolloutRefWorker),
        config=example_config,
        role="rollout"
    )
    example_resource_pool = RayResourcePool(
        process_on_nodes=[config.trainer.example_n_gpus_per_node] * config.trainer.example_nnodes,
        max_colocate_count=example_n_cpus_per_gpu  # This sets CPUs per GPU
    )
    example_wg = RayWorkerGroup(
        resource_pool=example_resource_pool,
        ray_cls_with_init=example_ray_cls,
        device_name=config.trainer.device,
    )
    example_wg.init_model()

    # Initialize Solution Generator Worker Group
    print("Initializing Solution Generator Worker Group...")
    # Build solution config by copying the base config structure
    solution_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    solution_config.model = config.solution_model
    solution_config.rollout = config.solution_rollout
    solution_config.actor = config.solution_actor
    # solution_config.ref = config.solution_rollout.ref
    
    # Get CPU count per GPU (default to 4 if not specified)
    solution_n_cpus_per_gpu = config.trainer.get("solution_n_cpus_per_gpu", 4)
    
    solution_ray_cls = RayClassWithInitArgs(
        cls=ray.remote(ActorRolloutRefWorker),
        config=solution_config,
        role="rollout"
    )
    solution_resource_pool = RayResourcePool(
        process_on_nodes=[config.trainer.solution_n_gpus_per_node] * config.trainer.solution_nnodes,
        max_colocate_count=solution_n_cpus_per_gpu  # This sets CPUs per GPU
    )
    solution_wg = RayWorkerGroup(
        resource_pool=solution_resource_pool,
        ray_cls_with_init=solution_ray_cls,
        device_name=config.trainer.device,
    )
    solution_wg.init_model()

    total_samples = len(dataset)
    config_batch_size = config.data.batch_size
    num_batch = -(-total_samples // config_batch_size)
    
    # Store outputs: [n_example_samples][n_solution_samples_per_example][n_data]
    example_output_lst = [[] for _ in range(n_example_samples)]
    solution_output_lst = [
        [[] for _ in range(n_solution_samples)]
        for _ in range(n_example_samples)
    ]

    max_prompt_length = config.data.get("max_prompt_length", 2048)
    truncation = config.data.get("truncation", "right")
    solution_prompt_template = config.data.get("solution_prompt_template", None)

    for batch_idx in range(num_batch):
        print(f"[{batch_idx + 1}/{num_batch}] Start to process batch.")
        
        batch_problem_texts = problem_texts_extracted[
            batch_idx * config_batch_size : (batch_idx + 1) * config_batch_size
        ]
        
        # === Step 1: Generate Examples ===
        print(f"[{batch_idx + 1}/{num_batch}] Generating examples...")
        
        # Format prompts for example generator
        example_prompts = format_problem_for_example_generator(batch_problem_texts)
        # breakpoint()
        # ray_pdb.set_trace()
        
        # Tokenize and prepare batch for example generator
        example_input_ids, example_attention_mask, example_position_ids, example_raw_prompt_ids = (
            process_generated_text_to_tensors(
                example_prompts,
                example_tokenizer,
                max_prompt_length,
                truncation
            )
        )
        
        example_batch_dict = {
            "input_ids": example_input_ids,
            "attention_mask": example_attention_mask,
            "position_ids": example_position_ids,
        }
        
        example_data = DataProto.from_dict(example_batch_dict)
        example_data_padded, example_pad_size = pad_dataproto_to_divisor(
            example_data, example_wg.world_size
        )
        
        # Generate examples n_example_samples times
        batch_example_texts_all = []
        for n_example in range(n_example_samples):
            print(f"  Generating example sample {n_example + 1}/{n_example_samples}")
            
            example_output_padded = example_wg.generate_sequences(example_data_padded)
            example_output = unpad_dataproto(example_output_padded, pad_size=example_pad_size)
            
            # Decode example responses
            batch_example_texts = []
            for i in range(len(example_output)):
                example_item = example_output[i]
                example_response_ids = example_item.batch["responses"]
                example_text = example_tokenizer.decode(example_response_ids, skip_special_tokens=True)
                batch_example_texts.append(example_text)
            
            example_output_lst[n_example].extend(batch_example_texts)
            batch_example_texts_all.append(batch_example_texts)
        
        # breakpoint()
        # ray_pdb.set_trace()

        # === Step 2: Generate Solutions conditioned on Examples ===
        print(f"[{batch_idx + 1}/{num_batch}] Generating solutions...")
        
        for n_example in range(n_example_samples):
            batch_example_texts = batch_example_texts_all[n_example]
            
            # Format prompts for solution generator (problem + example)
            solution_prompts = [
                format_conditioned_prompt(problem, example, solution_prompt_template)
                for problem, example in zip(batch_problem_texts, batch_example_texts, strict=True)
            ]
            # breakpoint()
            # ray_pdb.set_trace()
            
            # Tokenize and prepare batch for solution generator
            solution_input_ids, solution_attention_mask, solution_position_ids, solution_raw_prompt_ids = (
                process_generated_text_to_tensors(
                    solution_prompts,
                    solution_tokenizer,
                    max_prompt_length,
                    truncation
                )
            )
            
            solution_batch_dict = {
                "input_ids": solution_input_ids,
                "attention_mask": solution_attention_mask,
                "position_ids": solution_position_ids,
            }
            
            solution_data = DataProto.from_dict(solution_batch_dict)
            solution_data_padded, solution_pad_size = pad_dataproto_to_divisor(
                solution_data, solution_wg.world_size
            )
            
            # Generate solutions n_solution_samples times for this example
            for n_solution in range(n_solution_samples):
                print(f"  Example {n_example + 1}/{n_example_samples}, "
                      f"Solution {n_solution + 1}/{n_solution_samples}")
                
                solution_output_padded = solution_wg.generate_sequences(solution_data_padded)
                solution_output = unpad_dataproto(solution_output_padded, pad_size=solution_pad_size)
                
                # Decode solution responses
                batch_solution_texts = []
                for i in range(len(solution_output)):
                    solution_item = solution_output[i]
                    solution_response_ids = solution_item.batch["responses"]
                    solution_text = solution_tokenizer.decode(
                        solution_response_ids, skip_special_tokens=True
                    )
                    batch_solution_texts.append(solution_text)
                
                solution_output_lst[n_example][n_solution].extend(batch_solution_texts)
    
    # === Step 3: Prepare output dataframe ===
    print("Preparing output dataframe...")
    
    # Convert example outputs from (n_example_samples, n_data) to (n_data, n_example_samples)
    example_output_array = np.array(example_output_lst, dtype=object)
    example_output_array = np.transpose(example_output_array, axes=(1, 0))
    dataset["examples"] = example_output_array.tolist()
    
    # Convert solution outputs from (n_example_samples, n_solution_samples, n_data) 
    # to (n_data, n_example_samples, n_solution_samples)
    solution_output_array = np.array(solution_output_lst, dtype=object)  # (n_ex, n_sol, n_data)
    solution_output_array = np.transpose(solution_output_array, axes=(2, 0, 1))  # (n_data, n_ex, n_sol)
    dataset["solutions"] = solution_output_array.tolist()
    
    # Optionally flatten solutions if only 1 example sample
    if n_example_samples == 1:
        # Convert from (n_data, 1, n_solution_samples) to (n_data, n_solution_samples)
        dataset["solutions"] = [sols[0] for sols in dataset["solutions"]]
        dataset["examples"] = [ex[0] for ex in dataset["examples"]]
    
    # === Step 4: Save output ===
    output_dir = os.path.dirname(config.data.output_path)
    makedirs(output_dir, exist_ok=True)
    dataset.to_parquet(config.data.output_path)
    
    print(f"Two-player generation complete! Output saved to: {config.data.output_path}")
    print(f"Generated {n_example_samples} example(s) per problem")
    print(f"Generated {n_solution_samples} solution(s) per example")


if __name__ == "__main__":
    main()

