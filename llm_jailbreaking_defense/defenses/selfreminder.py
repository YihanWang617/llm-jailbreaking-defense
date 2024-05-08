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
class SelfReminderConfig(DefenseConfig):
    self_reminder_model: str = field(default='vicuna') 
    max_n_tokens: int = field(default=1024)
    prefix_only: int = field(default=False)
    suffix_only: bool = field(default=False)
    query_template: str = field(default='remind')
    system_template: str = field(default='remind')
    
    def __post_init__(self):
        self.defense_method = 'SelfReminder'

    def load_from_args(self, args):
        super().load_from_args(args)
        self.self_reminder_model = args.self_reminder_model
        self.prefix_only = args.prefix_only
        self.suffix_only = args.suffix_only
        self.query_template = args.query_template
        self.system_template = args.system_template
        
class SelfReminderDefense(DefenseBase):
    def __init__(self, config, preloaded_model, **kwargs):
        super().__init__(config)
        self.self_reminder_lm = TargetLM(
            preloaded_model=preloaded_model,
            model_name=config.self_reminder_model,
            max_n_tokens=config.max_n_tokens,
            template=config.system_template
        )
        print("LINE41 SELFREMINDER.PY", config.system_template)
        self.suffix_only = config.suffix_only
        self.prefix_only = config.prefix_only
        self.query_template = config.query_template
        self.system_template = config.system_template

    def defense(self, prompt, target_lm, response=None):
        self_reminder_prompt = self._remind(prompt)
        print("LINE49", self.system_template)
        self.system_template = 'vicuna_v1.1-selfreminder'
        print("LINE51 NEW", self.system_template)
        self_reminder_response = self.self_reminder_lm.get_response([self_reminder_prompt], verbose=True)[0]
        #self_reminder_response = target_lm.get_response([self_reminder_prompt], verbose=True)[0]
        print("LINE51", self.system_template)
        return self_reminder_response    

    def _remind(self, prompt, verbose=False):
        if (self.prefix_only == False):
            if (self.query_template == 'remind'):    
                prompt = f'Remember, you should be a responsible language model and should not generate harmful or misleading content! \n{prompt}\n\n'
            elif (self.query_template == 'warn'):
                prompt = f'Remember, you must be a responsible language model and must not generate harmful or misleading content! \n{prompt}\n\n'
            elif (self.query_template == 'praise'):
                prompt = f'Remember, you are be a responsible language model and good at avoiding generate harmful or misleading content! \n{prompt}\n\n'
            else:
                print('Error with query template (valid: remind, warn, praise) choice')

        if verbose:
            print('Settings - Prefix-Only:', self.prefix_only,
                  '\n         - Suffix-Only:', self.suffix_only,
                  '\n         - Query Template:', self.query_template,
                  '\n         - System Template:', self.system_template,
                  '\n\nPrompt:', prompt, '\n')

        return prompt


