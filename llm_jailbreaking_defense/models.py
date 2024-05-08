# MIT License

# Copyright (c) 2023-2024, Yihan Wang, Zhouxing Shi, Andrew Bai
# Copyright (c) 2023 PAIR Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Wrap targetLM with defense
The attacker has access to the defense strategy
"""
from packaging import version
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
import fastchat
from fastchat.model import get_conversation_template
from fastchat.conversation import (Conversation, SeparatorStyle,
                                   register_conv_template, get_conv_template
                                   )
from llm_jailbreaking_defense.language_models import GPT, Claude, HuggingFace


full_model_dict = {
    "gpt-4": {
        "path": "gpt-4",
        "template": "gpt-4"
    },
    "gpt-3.5-turbo": {
        "path": "gpt-3.5-turbo-0613",
        "template": "gpt-3.5-turbo"
    },
    "gpt-3.5-turbo-0301": {
        "path":"gpt-3.5-turbo-0301",
        "template":"gpt-3.5-turbo"
    },
    "gpt-3.5-turbo-0613": {
        "path":"gpt-3.5-turbo-0613",
        "template":"gpt-3.5-turbo"
    },
    "vicuna": {
        "path": "lmsys/vicuna-13b-v1.5",
        "template": "vicuna_v1.1"
    },
    "vicuna-13b-v1.5": {
        "path": "lmsys/vicuna-13b-v1.5",
        "template": "vicuna_v1.1"
    },
    "llama-2": {
        "path": "meta-llama/Llama-2-13b-chat-hf",
        "template": "llama-2"
    },
    "llama-2-13b": {
        "path": "meta-llama/Llama-2-13b-chat-hf",
        "template": "llama-2"
    },
    "claude-instant-1": {
        "path": "claude-instant-1",
        "template": "claude-instant-1"
    },
    "claude-2": {
        "path": "claude-2",
        "template": "claude-2"
    }
}


def conv_template(template_name):
    if template_name == "llama-2-new":
        # For compatibility with GCG
        template = get_conv_template(template_name)
    else:
        template = get_conversation_template(template_name)
    if template.name.startswith("llama-2"):
        template.sep2 = template.sep2.strip()
    return template


def load_indiv_model(model_name, max_memory=None, load_in_8bit=True):
    model_path, template = get_model_path_and_template(model_name)
    if model_name.startswith("gpt-"):
        lm = GPT(model_name)
    elif model_name.startswith("claude-"):
        lm = Claude(model_name)
    else:
        if load_in_8bit:
            if max_memory is not None:
                max_memory = {i: f"{max_memory}MB"
                            for i in range(torch.cuda.device_count())}

            model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True, device_map="auto",
                    load_in_8bit=True,
                    max_memory=max_memory).eval()
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,device_map="auto").eval()

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False
        )
        if "llama-2" in model_path.lower():
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.padding_side = "left"
        if "vicuna" in model_path.lower():
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token

        lm = HuggingFace(model_name, model, tokenizer)

    return lm, template


def register_model_path_and_template(model_name, model_path, model_template):
    full_model_dict[model_name] = {
        "path": model_path,
        "template": model_template
    }


def register_modified_llama_template():
    print("Using fastchat 0.2.20. Removing the system prompt for LLaMA-2.")
    register_conv_template(
        Conversation(  # pylint: disable=unexpected-keyword-arg
            name="llama-2-new",
            system="<s>[INST] <<SYS>>\n\n<</SYS>>\n\n",
            roles=("[INST]", "[/INST]"),
            messages=(),
            offset=0,
            sep_style=SeparatorStyle.LLAMA2,
            sep=" ",
            sep2=" </s><s>",
            stop_token_ids=[2],
        ),
        override=True
    )
    template = "llama-2-new"
    print("Using LLaMA-2 with fastchat < 0.2.24. "
          f"Template changed to `{template}`.")
    return template


def get_model_path_and_template(model_name):
    path = full_model_dict[model_name]["path"]
    template = full_model_dict[model_name]["template"]

    if template == "llama-2" and version.parse(
            fastchat.__version__) < version.parse("0.2.24"):
        template = register_modified_llama_template()

    return path, template


class TargetLM():
    """
    Base class for target language models.

    Generates responses for prompts using a language model.
    The self.model attribute contains the underlying generation model.
    """
    def __init__(
        self,
        model_name: str = None,
        max_n_tokens: int = 300,
        max_memory: int = None,
        temperature: float = 0.0,
        top_p: float = 1,
        preloaded_model: object = None,
        batch_size: int = 1,
        add_system_prompt: bool = True,
        template: str = None,
        quantization_config: BitsAndBytesConfig = None  
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_n_tokens = max_n_tokens
        self.top_p = top_p
        self.batch_size = batch_size
        self.add_system_prompt = add_system_prompt
        self.template = template
        self.quantization_config = quantization_config 

        assert model_name is not None or preloaded_model is not None
        if preloaded_model is None:
            self.model, self.template = load_indiv_model(
                model_name, max_memory=max_memory, load_in_8bit=quantization_config.load_in_8bit) # Use the quantization configuration
        else:
            self.model = preloaded_model
            assert template is not None or model_name is not None
            if self.template is None:
                _, self.template = get_model_path_and_template(model_name)

    def get_response(self, prompts_list, verbose=True, **kwargs):        
        batch_size = len(prompts_list)
        convs_list = [conv_template(self.template) for _ in range(batch_size)]
        full_prompts = []
        for conv, prompt in zip(convs_list, prompts_list):
            if isinstance(prompt, str):
                conv.append_message(conv.roles[0], prompt)
            elif isinstance(prompt, list): # handle multi-round conversations
                for i, p in enumerate(prompt):
                    conv.append_message(conv.roles[i % 2], p)
            else:
                raise NotImplementedError

            if self.model_name is not None and "gpt" in self.model_name:
                # Openai does not have separators
                full_prompts.append(conv.to_openai_api_messages())
            else:
                conv.append_message(conv.roles[1], None)
                full_prompts.append(conv.get_prompt())

        if not self.add_system_prompt:
            remove_system_prompts_pap(self, full_prompts)

        if verbose:
            print(f"Calling the TargetLM with {len(full_prompts)} prompts")
        outputs_list = []
        for i in tqdm(range((len(full_prompts)-1) // self.batch_size + 1),
                  desc="Target model inference on batch: ",
                  disable=(not verbose)):
            
            # Get the current batch of inputs
            batch = full_prompts[i * self.batch_size:(i+1) * self.batch_size]
          
        
            # Run a forward pass through the LLM for each perturbed copy
            batch_outputs = self.model.batched_generate(
                batch,
                max_n_tokens=self.max_n_tokens,
                temperature=self.temperature,
                top_p=self.top_p)
            
            outputs_list.extend(batch_outputs)
        return outputs_list

    def evaluate_log_likelihood(self, prompt, response):
        return self.model.evaluate_log_likelihood(prompt, response)


class DefendedTargetLM:
    def __init__(self, target_model, defense):
        self.target_model = target_model
        self.defense = defense

    def get_response(self, prompts_list, responses_list=None, verbose=False):
        if responses_list is not None:
            assert len(responses_list) == len(prompts_list)
        else:
            responses_list = [None for _ in prompts_list]
        defensed_response = [
            self.defense.defense(
                prompt, self.target_model, response=response
            )
            for prompt, response in zip(prompts_list, responses_list)
        ]
        return defensed_response

    def evaluate_log_likelihood(self, prompt, response):
        return self.target_model.evaluate_log_likelihood(prompt, response)


def remove_system_prompts_pap(target_lm, full_prompts):
    """Remove system prompts for PAP.

    This is mainly used for reproducing exisitng attacks such as PAP which
    does not add system prompts.
    """
    print("Removing the system prompt. (Only for running PAP!)")
    if "gpt" in target_lm.model_name:
        full_prompts = [[a for a in message if a["role"] != "system"]
                            for message in full_prompts]
    elif "llama" in target_lm.model_name:
        for i, prompt in enumerate(full_prompts):
            assert prompt.startswith("[INST]")
            assert prompt.endswith("[/INST]")
            # Just for consistency with PAP. Do not use for other attacks.
            full_prompts[i] = (
                target_lm.model.tokenizer.bos_token +
                f"[INST] <<SYS>>\n{None}\n<</SYS>>\n\n"
                f"{prompt[0][7:-8]} [/INST]"
            )
    else:
        raise NotImplementedError
