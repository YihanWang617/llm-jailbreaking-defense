from llm_jailbreaking_defense.judges.keyword import GCGKeywordMatchingJudge, KeywordMatchingJudge
from llm_jailbreaking_defense.judges.openai_policy_judge import OpenAIPolicyGPTJudge
from llm_jailbreaking_defense.judges.pair_judge import PAIRGPTJudge
from llm_jailbreaking_defense.judges.quality_judge import QualityGPTJudge
from llm_jailbreaking_defense.judges.no_judge import NoJudge


judge_dict={
    "quality": QualityGPTJudge,
    "openai_policy": OpenAIPolicyGPTJudge,
    "pair": PAIRGPTJudge,
    "no-judge": NoJudge,
    "matching": KeywordMatchingJudge,
    "gcg_matching": GCGKeywordMatchingJudge
}


def load_judge(judge_name, goal, **kwargs):
    if judge_name in judge_dict:
        judge = judge_dict[judge_name](goal, **kwargs)
    else:
        judge_name, model_name = judge_name.split("@")
        if judge_name in judge_dict:
            judge = judge_dict[judge_name](goal, model_name, **kwargs)
        else:
            raise NotImplementedError
    return judge
