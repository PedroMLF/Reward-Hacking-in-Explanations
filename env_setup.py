import os
from dotenv import load_dotenv

def setup_env(use_temp):
    load_dotenv()

    if use_temp:
        os.environ["HF_MODELS_CACHE_DIR"] = os.environ.get("HF_MODELS_CACHE_DIR_TMP")
    else:
        os.environ["HF_MODELS_CACHE_DIR"] = os.environ.get("HF_MODELS_CACHE_DIR")
    os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN")
    os.environ["HF_DATASETS_CACHE"] = os.environ["HF_MODELS_CACHE_DIR"]
    os.environ["HF_HOME"] = os.environ["HF_MODELS_CACHE_DIR"]
    os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_MODELS_CACHE_DIR"]
    os.environ["TOKENIZERS_PARALLELISM"] = "false"