import openai
import anthropic
import os
import time
import torch
import gc
from typing import Dict, List


class LanguageModel():
    def __init__(self, model_name):
        self.model_name = model_name

    def batched_generate(self, prompts: List, max_n_tokens: int, temperature: float):
        """
        Generates responses for a batch of prompts using a language model.
        """
        raise NotImplementedError

    def evaluate_log_likelihood(self, prompt, output):
        """
        Evaluate the per-token average likelihood of P(output|prompt).
        """
        print("Warning: `evaluate_log_likelihood` is not implemented "
              f"for model class {type(self)}. Returning 0 by default.")
        return 0


class HuggingFace(LanguageModel):
    def __init__(self, model_name, model, tokenizer):
        super().__init__(model_name)
        self.model = model
        self.tokenizer = tokenizer
        self.eos_token_ids = [self.tokenizer.eos_token_id]

    def batched_generate(self,
                         prompts: List,
                         max_n_tokens: int,
                         temperature: float,
                         top_p: float = 1.0):
        inputs = self.tokenizer(prompts, return_tensors='pt', padding=True)
        inputs = {k: v.to(self.model.device.index) for k, v in inputs.items()}

        # Batched generation
        if temperature > 0:
            do_sample = True
        else:
            do_sample = False
            top_p = temperature = 1  # To prevent warning messages
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_n_tokens,
            eos_token_id=self.eos_token_ids,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )

        # If the model is not an encoder-decoder type, slice off the input tokens
        if not self.model.config.is_encoder_decoder:
            output_ids = output_ids[:, inputs["input_ids"].shape[1]:]

        # Batch decoding
        outputs_list = self.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True)

        for key in inputs:
            inputs[key].to('cpu')
        output_ids.to('cpu')
        del inputs, output_ids
        gc.collect()
        torch.cuda.empty_cache()

        return outputs_list

    def extend_eos_tokens(self):
        # Add closing braces for Vicuna/Llama eos when using attacker model
        self.eos_token_ids.extend([
            self.tokenizer.encode("}")[1],
            29913,
            9092,
            16675])

    def evaluate_log_likelihood(self, prompt, output):
        inputs = [prompt + output]
        input_ids = self.tokenizer(
            inputs, padding=True, return_tensors="pt"
        ).input_ids.to(self.model.device)
        prompt_ids = self.tokenizer(
            [prompt], padding=True, return_tensors="pt"
        ).input_ids.to(self.model.device)
        prompt_length = prompt_ids.shape[-1]
        input_ids = input_ids[:,:prompt_length + 150]

        outputs = self.model(input_ids)
        probs = torch.log_softmax(outputs.logits, dim=-1).detach()

        # collect the probability of the generated token -- probability
        # at index 0 corresponds to the token at index 1
        probs = probs[:, prompt_length-1:-1, :]
        target_ids = input_ids[:, prompt_length:]
        gen_probs = torch.gather(probs, 2, target_ids[:, :, None]).squeeze(-1)

        return gen_probs.mean()


class GPT(LanguageModel):
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 5
    API_TIMEOUT = 20
    openai.api_key = os.getenv("OPENAI_API_KEY")

    def generate(self,
                 conv: List[Dict],
                 max_n_tokens: int,
                 temperature: float,
                 top_p: float):
        """
        Args:
            conv: List of dictionaries, OpenAI API format
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        """
        output = self.API_ERROR_OUTPUT
        for _ in range(self.API_MAX_RETRY):
            try:
                response = openai.ChatCompletion.create(
                    model = self.model_name,
                    messages = conv,
                    max_tokens = max_n_tokens,
                    temperature = temperature,
                    top_p = top_p,
                    request_timeout = self.API_TIMEOUT,
                )
                output = response["choices"][0]["message"]["content"]
                break
            except openai.error.OpenAIError as e:
                print(type(e), e)
                time.sleep(self.API_RETRY_SLEEP)

            time.sleep(self.API_QUERY_SLEEP)
        return output

    def batched_generate(self,
                         prompts: List[List[Dict]],
                         max_n_tokens: int,
                         temperature: float,
                         top_p: float = 1.0,):
        return [self.generate(prompt, max_n_tokens, temperature, top_p)
                for prompt in prompts]


class Claude(LanguageModel):
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 1
    API_MAX_RETRY = 5
    API_TIMEOUT = 20
    API_KEY = os.getenv("ANTHROPIC_API_KEY")

    def __init__(self, model_name):
        super().__init__(model_name)
        self.model= anthropic.Anthropic(api_key=self.API_KEY)

    def generate(self,
                 conv: List,
                 max_n_tokens: int,
                 temperature: float,
                 top_p: float):
        """
        Args:
            conv: List of conversations
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        """
        output = self.API_ERROR_OUTPUT
        for _ in range(self.API_MAX_RETRY):
            try:
                completion = self.model.completions.create(
                    model=self.model_name,
                    max_tokens_to_sample=max_n_tokens,
                    prompt=conv,
                    temperature=temperature,
                    top_p=top_p
                )
                output = completion.completion
                break
            except anthropic.APIError as e:
                print(type(e), e)
                time.sleep(self.API_RETRY_SLEEP)
            time.sleep(self.API_QUERY_SLEEP)
        return output

    def batched_generate(self,
                         prompts: List[List[Dict]],
                         max_n_tokens: int,
                         temperature: float,
                         top_p: float = 1.0):
        return [self.generate(prompt, max_n_tokens, temperature, top_p)
                for prompt in prompts]
