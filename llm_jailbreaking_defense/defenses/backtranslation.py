import math
from dataclasses import dataclass, field
from llm_jailbreaking_defense.defenses.base import DefenseBase, DefenseConfig
from llm_jailbreaking_defense.judges import check_rejection
from llm_jailbreaking_defense.models import TargetLM


@dataclass
class BacktranslationConfig(DefenseConfig):
    threshold: float = field(default=-2.0)
    infer_model: str = field(default='vicuna')
    infer_lm_length: int = field(default=None)
    new_response_length: int = field(default=None)
    defense_lm_max_memory: float = field(default=None)
    return_new_response_anyway: bool = field(default=False)

    def __post_init__(self):
        self.defense_method = "backtranslation"

    def load_from_args(self, args):
        super().load_from_args(args)
        self.threshold = args.backtranslation_threshold
        self.infer_model = args.backtranslation_infer_model
        self.infer_lm_length = args.target_max_n_tokens
        self.new_response_length = args.backtranslation_new_response_length
        self.defense_lm_max_memory = args.max_memory
        self.return_new_response_anyway = args.return_new_response_anyway


class BackTranslationDefense(DefenseBase):
    def __init__(self, config, preloaded_model, **kwargs):
        super().__init__(config)
        print(f'Using threshold {config.threshold} for backtranslation')
        infer_lm = TargetLM(model_name=config.infer_model,
                            max_n_tokens=config.infer_lm_length,
                            max_memory=config.defense_lm_max_memory,
                            preloaded_model=preloaded_model)
        self.infer_lm = infer_lm
        self.threshold = config.threshold
        self.return_new_response_anyway = config.return_new_response_anyway
        self.new_response_length = config.new_response_length

    def defense(self, prompt, target_lm, response=None):
        # I'm testing to see why self_reminder_lm has the batched_generate issue
        #test = self.infer_lm.get_response([prompt], verbose=self.verbose)[0]
        #print("LINE45 BACKTRANSLATION.PY DEBUG", test)
        if self.verbose:
            print("Adversarial harmful prompt: \n", prompt)
        if response is None:
            response = target_lm.get_response([prompt], verbose=self.verbose)[0]

        if check_rejection(response, exclude_lack_of_info=False):
            return "I'm sorry, but I cannot assist with that request."

        possible_prompt = self._infer_question(response)
        print("LINE56 BACKTRANSLATION.PY DEBUG", possible_prompt)
        is_reasonable = self._filter_question_with_likelihood(
            possible_prompt, response)
        if not is_reasonable:
            return response

        if self.verbose:
            print("Backtranslated harmful prompt: \n", possible_prompt)
        new_response = target_lm.get_response(
            [possible_prompt],
            max_n_tokens=self.new_response_length,
            verbose=self.verbose
        )[0]
        if self.verbose:
            print("New response with backtranslated prompt: \n", new_response)

        if self.return_new_response_anyway:
            return new_response

        if not check_rejection(new_response):
            return response

        return "I'm sorry, but I cannot assist with that request."

    def _infer_question(self, response):
        # PROBLEM: it looks like it never goes to _infer_question with the basic response
        # ... this means that the infer_lm issue is maybe overlooked?
        print("LINE80 BACKTRANSLATION.PY DEBUG")
        infer_prompt = (
            'Please guess the user\'s request the AI assistant answers in the '
            'following response from an AI assistant. '
            'If there are multiple possible requests, give the most harmful one: '
            f'"{response}".\n\n'
            'Output the inferred request following this format: "Request: [[request]]". '
            'For example, "Request: [[Create a cake recipe.]]".'
        )
        if self.verbose:
            print('Asking the model to infer the prompt:')
            print(infer_prompt)
        output = self.infer_lm.get_response([infer_prompt], verbose=self.verbose)[0]
        print("LINE89 BACKTRANSLATION.PY DEBUG", output)
        # I'm testing to see why self_reminder_lm has the batched_generate issue
        if ':' not in output:
            if self.verbose:
                print(f"Parse error with output: {output}")
            return ""
        ret = output.split(':')[-1].split("\n")[0].strip().strip(']').strip('[')
        if self.verbose:
            print('Inferred prompt:', ret)
        return ret

    def _filter_question_with_likelihood(self, prompt, response):
        if self.threshold > -math.inf:
            avg_log_likelihood = self.infer_lm.evaluate_log_likelihood(
                prompt, response)
            if self.verbose:
                print(f"average log likelihood: {avg_log_likelihood}")
            return avg_log_likelihood > self.threshold
        else:
            return True
