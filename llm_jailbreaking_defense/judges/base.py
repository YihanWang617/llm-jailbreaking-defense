class JudgeBase:
    def __init__(self, goal, model_name=None, **kwargs):
        self.judge_model = None
        self.goal = goal
        self.model_name = model_name

    def score(self, prompt_list, response_list):
        raise NotImplementedError
