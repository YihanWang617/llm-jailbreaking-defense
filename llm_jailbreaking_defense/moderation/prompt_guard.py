from llm_jailbreaking_defense.moderation.base import Moderation

from transformers import pipeline


class PromptGuard(Moderation):
    def __init__(self, device="cuda"):
        super().__init__(device)
        self.classifier = pipeline(
            "text-classification", model="meta-llama/Prompt-Guard-86M",
            device=device)

    def predict(self, prompt: str):
        assert isinstance(prompt, str)
        ret_raw = self.classifier(prompt)[0]
        if ret_raw['label'] == 'BENIGN':
            ret = {
                'type': None,
                'score': 1 - ret_raw['score']
            }
        else:
            ret = {
                'type': ret_raw['label'],
                'score': ret_raw['score']
            }
        return ret
