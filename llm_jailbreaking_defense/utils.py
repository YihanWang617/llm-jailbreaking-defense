def remove_system_prompts_pap(target_lm, full_prompts):
    """Remove system prompts for PAP.

    This is mainly used for reproducing exisitng attacks such as PAP which
    does not add system prompts.
    """
    print("Removing the system prompt. (Only for running PAP!)")
    if "gpt" in target_lm.model_name:
        full_prompts = [[a for a in message if a["role"] != "system"]
                            for message in full_prompts]
        return full_prompts
    elif "llama" in target_lm.model_name:
        for i, prompt in enumerate(full_prompts):
            assert prompt.startswith("[INST]")
            assert prompt.endswith("[/INST]")
            # Just for consistency with PAP. Do not use for other attacks.
            full_prompts[i] = (
                target_lm.model.tokenizer.bos_token +
                f"[INST] <<SYS>>\n{None}\n<</SYS>>\n\n"
                f"{prompt[0][7:-8]} [/INST]"
            )
        return full_prompts
    else:
        raise NotImplementedError
