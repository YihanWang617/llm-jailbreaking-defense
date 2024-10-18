from dataclasses import dataclass, field

@dataclass
class DefenseConfig:
    defense_method: str = field(
        default='None'
    )
    verbose: bool= field(
        default=False
    )

    def load_from_args(self, args):
        self.verbose = args.verbose


class DefenseBase():
    def __init__(self, config, preloaded_model=None, **kwargs):
        self.defense_method = config.defense_method
        self.verbose = config.verbose
        self.preloaded_model = preloaded_model

    def defense(self, prompt, target_lm, response=None):
        if response is None:
            response = target_lm.get_response([prompt], verbose=self.verbose)[0]
        return response
