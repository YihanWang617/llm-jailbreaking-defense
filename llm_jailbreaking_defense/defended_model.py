# MIT License

# Copyright (c) 2023-2024, Yihan Wang, Zhouxing Shi, Andrew Bai

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


class DefendedTargetLM:
    def __init__(self, target_model, defense):
        self.target_model = target_model
        self.defense = defense

    def get_response(self, prompts_list, responses_list=None, verbose=False):
        if responses_list is not None:
            assert len(responses_list) == len(prompts_list)
        else:
            responses_list = [None for _ in prompts_list]
        defensed_response = [
            self.defense.defense(
                prompt, self.target_model, response=response
            )
            for prompt, response in zip(prompts_list, responses_list)
        ]
        return defensed_response

    def evaluate_log_likelihood(self, prompt, response):
        return self.target_model.evaluate_log_likelihood(prompt, response)
