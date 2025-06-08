# llm_model.py
from transformers import AutoModelForCausalLM

def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype="auto",         # 자동으로 float16 등으로 설정
        device_map="auto"           # GPU 자동 할당
    )
    return model
