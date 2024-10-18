from setuptools import setup, find_packages

VERSION = '0.0.2'
DESCRIPTION = 'A package for jailbreaking defenses'

setup(
    name="llm_jailbreaking_defense",
    version=VERSION,
    author="Yihan Wang, Zhouxing Shi, Andrew Bai, Cho-Jui Hsieh",
    author_email="wangyihan617@gmail.com, zshi@cs.ucla.edu, andrewbai@ucla.edu, chohsieh@cs.ucla.edu",
    description=DESCRIPTION,
    packages=find_packages(),
    install_requires=[
        "protobuf>=3.19.5,<5.0.0.dev0",
        "torch>=2.0",
        "sentencepiece",
        "transformers",
        "fschat>=0.2.33",
        "pandas>=2.1",
        "seaborn>=0.13",
        "matplotlib>=3.8.2",
        "numpy>=1.26,<2",
        "datasets>=2.14.0",
        "evaluate>=0.4.1",
        "anthropic>=0.7.6",
        "openai>=1.40",
        "chardet>=5.2.0",
        "accelerate>=0.24",
        "bitsandbytes>=0.41",
        "scipy>=1.11",
        "ml_collections>=0.1.1",
        "nltk>=3.8",
        "aiohttp>=3.8.1",
        "fsspec[http]<=2024.2.0,>=2023.1.0"
    ],
    keywords=["Large Language Model", "LLM", "jailbreaking", "robustness"],
)
