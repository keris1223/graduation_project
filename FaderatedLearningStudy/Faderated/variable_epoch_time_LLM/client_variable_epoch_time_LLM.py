import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import socket
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import time
from utils import send_pickle, recv_pickle
from transformers import AutoTokenizer
from llm_model import load_model


SERVER_IP = '112.168.69.70'
PORT = 5000
NUM_ROUNDS = 10

total_train_time = 0.0
total_comm_time_to_server = 0.0
total_comm_time_from_server = 0.0

# 모델 및 학습 준비
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')
model.to(device)
model.train()

optimizer = optim.AdamW(model.parameters(), lr=1e-5)

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((SERVER_IP, PORT))

text = "User: Hello\\nAssistant: Hi, how can I help you?"
# 서버 연결


for round_num in range(1, NUM_ROUNDS + 1):
    print(f"\nRound {round_num}/{NUM_ROUNDS} 시작")

    if round_num > 1:
        try:
            # 서버에서 평균 모델과 local_epochs 수신
            start = time.perf_counter()
            payload = recv_pickle(client_socket)
            end = time.perf_counter()
            print(f"[Round {round_num}] 서버 → 클라 수신 시간: {end - start:.4f}s")
            total_comm_time_from_server += end - start

            avg_state_dict = payload['model']
            local_epochs = payload.get('local_epochs', 1)
            model.load_state_dict(avg_state_dict)
            print(f"[Round {round_num}] 서버에서 받은 Epoch 수: {local_epochs}")

        except Exception as e:
            print(f"[Round {round_num}] 수신 실패: {e}")
            break
    else:
        local_epochs = 1  # 첫 라운드는 고정

    # 로컬 학습
    train_start = time.perf_counter()
    cumulative_loss = 0.0
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    labels = inputs["input_ids"].clone()

    for epoch in range(local_epochs):
        print(f"Epoch {epoch + 1}/{local_epochs}")
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        cumulative_loss += loss.item()
        
    train_end = time.perf_counter()
    print(f"[Round {round_num}] 로컬 학습 시간: {train_end - train_start:.4f}s")
    total_train_time += train_end - train_start

    # 평균 loss 계산
    avg_loss = cumulative_loss / local_epochs
    print(f"[Round {round_num}] 평균 Loss: {avg_loss:.4f}")

    # 모델 + loss 전송
    try:
        start = time.perf_counter()
        send_pickle(client_socket, {
            'model': model.state_dict(),
            'metric': {
                'loss': avg_loss,
                'round': round_num
            }
        })
        end = time.perf_counter()
        print(f"[Round {round_num}] 클라→서버 전송 시간: {end - start:.4f}s")
        total_comm_time_to_server += end - start

        if round_num == NUM_ROUNDS:
            send_pickle(client_socket, {"done": True})
            print("[Client] done 메시지 전송 완료")

    except Exception as e:
        print(f"[Round {round_num}] 전송 실패: {e}")
        break

# 종료
client_socket.close()
print("\n전체 라운드 완료")
print(f"총 학습 시간: {total_train_time:.4f}s")
print(f"총 통신 지연 시간(클라이언트→서버): {total_comm_time_to_server:.4f}s")
print(f"총 통신 지연 시간(서버→클라이언트): {total_comm_time_from_server:.4f}s")
