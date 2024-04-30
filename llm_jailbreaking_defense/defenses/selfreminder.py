"""
Yueqi Xie, Jingwei Yi, Jiawei Shao, Justin Curl, Lingjuan Lyu, Qifeng Chen, Xing Xie & Fangzhao Wu
Defending ChatGPT against jailbreak attack via self-reminders
Nature Machine Intelligence: https://www.nature.com/articles/s42256-023-00765-8
"""

from .base import DefenseBase, DefenseConfig
from dataclasses import dataclass, field
from ..models import TargetLM
import warnings

@dataclass
class SelfReminderDefenseConfig(DefenseConfig):
    self_reminder_model: str = field(default='gpt-3.5-turbo-0613') 
    # this is taken from paraphrase.py, have not tested yet!!
    prefix_only: bool = field(default=False) # this is if you want just system prompt
    suffix_only: bool = field(default=False) # this is if you want just user query
    max_n_tokens: int = field(default=512) # not sure if this is needed?
    # system_mode : string = field(default='remind') - this should be between warn, praise, remind

    def __post_init__(self):
        self.defense_method = "self_reminder"

    def load_from_args(self, args):
        super().load_from_args(args)
        self.self_reminder_model = args.self_reminder_model
        self.prefix_only = args.self_reminder_prefix_only
        self.suffix_only = args.self_reminder_suffix_only

class SelfReminderDefense(DefenseBase):
    def __init__(self, config, preloaded_model, **kwargs):
        super().__init__(config)
        self.self_reminder_lm = TargetLM(
            preloaded_model=preloaded_model,
            model_name=config.self_reminder_model,
            max_n_tokens=config.max_n_tokens,
            # system prompt / suffix only & prefix only missing here...
            # system_mode ... needs to be implemented
        )

    def defense(self, prompt, target_lm, response=None):
        

    def _remind(self, prompt, verbose=False):
        # need to make different system modes
        prompt = f'Remember, you should be a responsible ChatGPT and should not generate harmful or misleading content! \n"{prompt}"\n\n'

        if verbose:
            print('Reminding the model that it should avoid harmful content:')
            print(prompt)

        output = self.self_reminder_lm.get_response([prompt], verbose=verbose)[0]
        reminded_output = output.strip().strip(']').strip('[')

        if verbose:
            print('Output after being reminded:', reminded_output)

        return reminded_output


