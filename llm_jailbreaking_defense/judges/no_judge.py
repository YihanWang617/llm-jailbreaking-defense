from llm_jailbreaking_defense.judges.base import JudgeBase


class NoJudge(JudgeBase):
    def __init__(self, goal, model_name=None, **kwargs):
        super().__init__(goal, model_name)
        self.judge_model = None

    def score(self, prompt_list, response_list):
        return [1 for _ in prompt_list]
