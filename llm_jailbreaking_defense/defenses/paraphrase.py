"""
Neel Jain, Avi Schwarzschild, Yuxin Wen, Gowthami Somepalli, John Kirchenbauer, Ping-yeh Chiang, Micah Goldblum, Aniruddha Saha, Jonas Geiping, and Tom Goldstein. 2023.
Baseline defenses for adversarial attacks against aligned language models.
arXiv preprint arXiv:2309.00614.
"""

from .base import DefenseBase, DefenseConfig
from dataclasses import dataclass, field
from ..models import TargetLM
import warnings


@dataclass
class ParaphraseDefenseConfig(DefenseConfig):
    paraphrase_model: str = field(default='gpt-3.5-turbo-0613')
    paraphrase_model_max_memory: float = field(default=None)
    max_n_tokens: int = field(default=1024)

    def load_from_args(self, args):
        super().load_from_args(args)
        self.paraphrase_model = args.paraphrase_model
        self.paraphrase_model_max_memory = args.max_memory

    def __post_init__(self):
        self.defense_method = "paraphrase_prompt"


class ParaphraseDefense(DefenseBase):
    def __init__(self, config, preloaded_model, **kwargs):
        super().__init__(config)
        self.paraphrase_lm = TargetLM(
            preloaded_model=preloaded_model,
            model_name=config.paraphrase_model,
            max_n_tokens=config.max_n_tokens,
            max_memory=config.defense_lm_max_memory,
            add_system_prompt=not config.no_system_prompt)

    def defense(self, prompt, target_lm, response=None):
        paraphrase_prompt = self._paraphrase(prompt)
        if "\n" in paraphrase_prompt:
            warnings.warn("""A \\n character is found in the output of paraphrase model and the content after \\n is removed.""")
            paraphrase_prompt = "\n".join(paraphrase_prompt.split('\n')[1:])
        paraphrase_response = target_lm.get_response(
            [paraphrase_prompt], verbose=self.verbose)[0]
        return paraphrase_response

    def _paraphrase(self, prompt, verbose=False):
        prompt = f'paraphrase the following paragraph: \n"{prompt}"\n\n'

        if verbose:
            print('Asking the model to paraphrase the prompt:')
            print(prompt)

        output = self.paraphrase_lm.get_response([prompt], verbose=verbose)[0]
        paraphrase_prompt = output.strip().strip(']').strip('[')

        if verbose:
            print('Paraphrased prompt:', paraphrase_prompt)

        return paraphrase_prompt
