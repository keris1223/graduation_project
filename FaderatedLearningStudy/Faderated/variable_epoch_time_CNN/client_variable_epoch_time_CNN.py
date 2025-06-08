import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import socket
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import BigCNN
import time
from utils import send_pickle, recv_pickle

SERVER_IP = '118.34.145.83'
PORT = 5000
NUM_ROUNDS = 10

total_train_time = 0.0
total_comm_time_to_server = 0.0
total_comm_time_from_server = 0.0

# 모델 및 학습 준비
model = BigCNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')
model.to(device)

transform = transforms.ToTensor()
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transform),
    batch_size=4, shuffle=True
)

optimizer = optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()
# 서버 연결
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((SERVER_IP, PORT))

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
            local_epochs = payload.get('local_epochs', 3)
            model.load_state_dict(avg_state_dict)
            print(f"[Round {round_num}] 서버에서 받은 Epoch 수: {local_epochs}")

        except Exception as e:
            print(f"[Round {round_num}] 수신 실패: {e}")
            break
    else:
        local_epochs = 3  # 첫 라운드는 고정

    # 로컬 학습
    train_start = time.perf_counter()
    model.train()
    cumulative_loss = 0.0
    total_batches = 0
    for epoch in range(local_epochs):
        print(f"Epoch {epoch + 1}/{local_epochs}")
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            cumulative_loss += loss.item()
            total_batches += 1
            break

              # 한 배치만 학습
    train_end = time.perf_counter()
    print(f"[Round {round_num}] 로컬 학습 시간: {train_end - train_start:.4f}s")
    total_train_time += train_end - train_start

    # 평균 loss 계산
    avg_loss = cumulative_loss / total_batches
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
