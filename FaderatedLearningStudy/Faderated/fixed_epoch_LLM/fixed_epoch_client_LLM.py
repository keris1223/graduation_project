import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import socket, torch, time, json, random
import torch.optim as optim
from transformers import AutoTokenizer
from utils import send_pickle, recv_pickle
from llm_model import load_model
from peft import get_peft_model_state_dict

SERVER_IP = '118.34.145.27'
PORT = 5000
NUM_ROUNDS = 50
BATCH_SIZE = 4
# FIXED_EPOCH = 3

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = load_model().to("cuda" if torch.cuda.is_available() else "cpu")
model.train()
optimizer = optim.AdamW(model.parameters(), lr=1e-5)

def load_prompts(path, tokenizer):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [f"User: {item['instruction']}\nAssistant: {item['output']}" for item in data]

prompts = load_prompts("../stanford_alpaca/alpaca_data.json", tokenizer)

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((SERVER_IP, PORT))

for round_num in range(1, NUM_ROUNDS + 1):
    print(f"\nRound {round_num} 시작")

    if round_num > 1:
        payload = recv_pickle(client_socket)
        model.load_state_dict(payload['model'], strict=False)
        local_epochs = payload.get('local_epochs',3)
        print(f"[Round {round_num}] 서버에서 모델 수신 완료")
    else:
        local_epochs = 5
    cumulative_loss = 0.0
    for epoch in range(local_epochs):
        epoch_loss = 0.0
        for _ in range(BATCH_SIZE):
            text = random.choice(prompts)
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(model.device)
            labels = inputs["input_ids"].clone()
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            epoch_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        avg_epoch_loss = epoch_loss / BATCH_SIZE
        cumulative_loss += avg_epoch_loss
        print(f"Epoch {epoch+1}/{local_epochs} - Loss: {loss.item():.4f}")

    avg_loss = cumulative_loss / local_epochs
    send_pickle(client_socket, {
        'model': get_peft_model_state_dict(model),
        'metric': {'loss': avg_loss, 'round': round_num}
    })

    if round_num == NUM_ROUNDS:
        send_pickle(client_socket, {"done": True})
        print("[Client] done 메시지 전송 완료")

client_socket.close()
