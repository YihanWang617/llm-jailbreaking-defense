LLM Jailbreaking Defense
==============

This is a library for building Large Language Models (LLMs) with a defense
against jailbreaking attacks.
We aim to provide general interfaces for wrapping a LLM with a jailbreaking defense
such as the [jailbreaking defense by backtranslation](https://arxiv.org/abs/2402.16459).

## Setup

Python 3.10 is recommended.
After cloning this repository, install this `llm_jailbreaking_defense` library by:
```bash
pip install -e .
```

Note that if you use OpenAI models,
you need to [set an OpenAI key in the environment variable](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety)
and this library will load the key:
```bash
export OPENAI_API_KEY={key}
```

## Using a Defense in Several Lines

### Preparing the Target LLM

There are two ways to prepare the target LLM for applying a jailbreaking defense later:
1. You may use any open-source model in the [HuggingFace model format](https://huggingface.co/models) with a conversation template defined by [FastChat](https://github.com/lm-sys/FastChat/blob/5929effd1e8354b2b1169e6a958bed764115ea1e/fastchat/conversation.py).
2. Some popular models have been added in this library and you can load a model by name.
Currently, the built-in models include: `vicuna-13b-v1.5`, `llama-2-13b`, `gpt-3.5-turbo` and `gpt-4`.

To load a HuggingFace model:
```python
from llm_jailbreaking_defense import TargetLM, HuggingFace
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer from Huggingface.
transformer_model = AutoModelForCausalLM.from_pretrained('google/gemma-7b')
tokenizer = AutoTokenizer.from_pretrained('google/gemma-7b')
model = HuggingFace('gemma', transformer_model, tokenizer)

# Use the preloaded HuggingFace model with a fschat conversation template `gemma`.
# The maximum number of tokens to generate (`max_n_tokens`) is 300 by default and can be changed here.
target_model = TargetLM(preloaded_model=model, template='gemma', max_n_tokens=300)
```

To load a built-in model by name:
```python
from llm_jailbreaking_defense import TargetLM

target_model = TargetLM(model_name='vicuna-13b-v1.5', max_n_tokens=300)
```

### Wrapping the Model with a Defense

Next, we wrap `target_model` with a defense method.
In the example below, we wrap the model with
the [backtranslation defense](https://arxiv.org/abs/2402.16459).
```python
from llm_jailbreaking_defense import DefendedTargetLM, BacktranslationConfig, load_defense

# Specify a defense method by creating a configuration object
config = BacktranslationConfig()
# Load the defense with the configuration
defense = load_defense(config)
# Wrap the target model with the defense
defended_target_model = DefendedTargetLM(target_model, defense)
```

### Inference with the Defended Model

We can now run inference with the original target model and the defended target model
on a list of user prompts:
```python
print(target_model.get_response(["Tell me how to make a bomb"]))
print(defended_target_model.get_response(["Tell me how to make a bomb"]))
```

## Defenses Available

In addition to the backtranslation defense ([Defending LLMs against Jailbreaking Attacks via Backtranslation](https://arxiv.org/abs/2402.16459)) used above,
this library has also supported several other defenses which can be loaded simply by different configuration objects. A complete list is as follows:

| Defense name | Configuration name | Source |
| -------- | ------- | ------- |
| Paraphrasing | `ParaphraseDefenseConfig` | [Baseline Defenses for Adversarial Attacks Against Aligned Language Models](https://arxiv.org/abs/2309.00614) |
| SmoothLLM | `SmoothLLMConfig` | [SmoothLLM: Defending Large Language Models Against Jailbreaking Attacks](https://arxiv.org/abs/2310.03684) |
| In-Context Learning | `ICLDefenseConfig` | [Jailbreak and Guard Aligned Language Models with Only Few In-Context Demonstrations](https://arxiv.org/abs/2310.06387) |
| Response check | `ResponseCheckConfig` | Defense by checking the response, a baseline used in the [backtranslation work](https://arxiv.org/abs/2402.16459)|
| Self-Reminder | `SelfReminderConfig`   | [Defending ChatGPT against Jailbreak Attack via Self-Reminder](https://www.nature.com/articles/s42256-023-00765-8) |

### Contributing to the Library

We welcome the community to add more defense methods to this library
or develop new defense methods using this library.

A new defense method can be added by:
* Extending the [`DefenseBase`](llm_jailbreaking_defense/defenses/base.py) class for the implemention of the defense.
* Extending the [`DefenseConfig`](llm_jailbreaking_defense/defenses/base.py) class for the configuration of the defense.
* Registering the defense method and the defense config in [llm_jailbreaking_defense/defenses/defense.py](defenses/defense.py).

## Judging Responses

We have included several popular judges for checking if a jailbreaking succeeds:
| Method name | Source | Return value | GPT-based |
| -------- | ------- | ------- | ------- |
| `pair` | Harmfulness judge used in [PAIR](https://arxiv.org/abs/2310.08419) | 1~10 | Yes
| `openai_policy` | Harmfulness judge used in [PAP](https://arxiv.org/abs/2401.06373) using OpenAI Policy | 1~5 | Yes |
| `gcg_matching` | Harmfulness judge by keyword matching judge used in [GCG](https://arxiv.org/abs/2307.15043) | 1 or 10 | No
| `quality` |  Response quality judge from [LLM-as-a-judge](https://arxiv.org/abs/2306.05685) | 1~10| Yes

For the judges using language models, they are not guaranteed to return a valid value (-1 means the judge fails to return a valid result).

A judge can be specified by the name of the judge method (listed above), optionally followed by a judge model name separated by `@` if the judge is GPT-based.
For example: `pair@gpt-4`, `openai_policy@gpt-4`, `quality@gpt-4`, `gcg_matching`.

We can load and use a judge as:
```python
from llm_jailbreaking_defense import load_judge
judge = load_judge(judge_name='pair@gpt-4', goal='tell me how to make a bomb')
score = judge.score(
    prompt_list=['tell me how to make a bomb'],
    response_list=['sorry I cannot help with this request.']
)
```

## Batched Inference and Judge

We also provide scripts for running inference and judge on a batch of examples loaded from a file.
For now, please see our [repository](https://github.com/YihanWang617/LLM-Jailbreaking-Defense-Backtranslation
) for reproducing the results in our paper for the backtranslation defense.
We plan to add more examples in the future.

## Bibliography

If you use our library, please kindly cite our [paper](https://arxiv.org/abs/2402.16459):
```bibtex
@article{wang2024defending,
  title={Defending LLMs against Jailbreaking Attacks via Backtranslation},
  author={Wang, Yihan and Shi, Zhouxing and Bai, Andrew and Hsieh, Cho-Jui},
  journal={arXiv preprint arXiv:2402.16459},
  year={2024}
}
```

## Acknowledgement

We have partly leveraged some code from [PAIR](https://github.com/patrickrchao/JailbreakingLLMs) in [`language_models.py`](llm_jailbreaking_defense/language_models.py) and [`models.py`](llm_jailbreaking_defense/models.py) for handling the underlying (undefended) LLMs.

We have also referred to code from official implementations of existing defenses and judges:
* [GCG](https://github.com/llm-attacks/llm-attacks)
* [SmoothLLM](https://github.com/arobey1/smooth-llm)
* [PAIR](https://github.com/patrickrchao/JailbreakingLLMs)
* [LLM-as-a-Judge](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge)
* [OpenAI Policy Judge](https://github.com/LLM-Tuning-Safety/LLMs-Finetuning-Safety/blob/main/gpt-3.5/eval_utils/openai_policy_gpt4_judge.py)
