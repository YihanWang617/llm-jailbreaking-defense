class Moderation():
    def __init__(self, device):
        self.device = device

    def predict(self, prompt):
        raise NotImplementedError

    def __call__(self, prompt):
        return self.predict(prompt)
