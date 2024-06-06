from .backtranslation import BackTranslationDefense, BacktranslationConfig
from .response_check import ResponseCheckDefense, ResponseCheckConfig
from .smoothllm import SmoothLLMDefense, SmoothLLMConfig
from .paraphrase import ParaphraseDefense, ParaphraseDefenseConfig
from .base import DefenseBase, DefenseConfig
from .ICL import ICLDefense, ICLDefenseConfig
from .semantic_smoothing import SemanticSmoothConfig, SemanticSmoothDefense
from .self_reminder import SelfReminderConfig, SelfReminderDefense


defense_dict = {
    'None': DefenseBase,
     None: DefenseBase,
    'SmoothLLM': SmoothLLMDefense,
    'backtranslation': BackTranslationDefense,
    'response_check': ResponseCheckDefense,
    'paraphrase_prompt': ParaphraseDefense,
    'SelfReminder': SelfReminderDefense,
    'ICL': ICLDefense,
    'SemanticSmooth': SemanticSmoothDefense
}

defense_config_dict = {
    'SmoothLLM': SmoothLLMConfig,
    'backtranslation': BacktranslationConfig,
    'response_check': ResponseCheckConfig,
    'paraphrase_prompt': ParaphraseDefenseConfig,
    'ICL': ICLDefenseConfig,
    'SelfReminder': SelfReminderConfig,
    'None': DefenseConfig,
    None: DefenseConfig,
    'SemanticSmooth': SemanticSmoothConfig
}


def args_to_defense_config(args):
    defense_method = args.defense_method
    if defense_method.startswith("backtranslation"):
        defense_method = "backtranslation"
    config = defense_config_dict[defense_method](args.defense_method)
    config.load_from_args(args)
    return config


def load_defense(defense_config, preloaded_model=None):
    defense_method = defense_config.defense_method
    if defense_method.startswith("backtranslation"):
        if 'threshold' in defense_method:
            threshold = float(defense_method.split('_')[-1])
            defense_method = "backtranslation"
            if threshold > 0:
                threshold = -threshold
            defense_config.backtranslation_threshold = threshold
        else:
            defense_config.backtranslation_threshold = -2.0

    return defense_dict[defense_method](defense_config, preloaded_model)
