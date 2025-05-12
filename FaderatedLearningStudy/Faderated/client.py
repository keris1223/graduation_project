import socket
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import BigCNN
import time
from utils import send_pickle, recv_pickle

SERVER_IP = '118.34.145.83'  # 예: '192.168.0.101'
PORT = 5000
NUM_ROUNDS = 10

total_train_time = 0.0
total_comm_time_to_server = 0.0
total_comm_time_from_server = 0.0

# 모델 및 학습
model = BigCNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')
model.to(device)

transform = transforms.ToTensor()
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transform),
    batch_size=8, shuffle=True
)

optimizer = optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((SERVER_IP, PORT))



for round_num in range(1, NUM_ROUNDS + 1):
    print(f"\nRound {round_num}/{NUM_ROUNDS} 시작")

    train_start = time.perf_counter()

    model.train()
    for epoch in range(3):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            break

    train_end = time.perf_counter()
    print(f"로컬 학습 시간: {train_end - train_start:.4f}s")
    total_train_time += train_end - train_start

    try:
        start = time.perf_counter()
        send_pickle(client_socket, model.state_dict())
        end = time.perf_counter()
        print(f"클라이언트->서버 전송 지연 시간: {end - start:.4f}초")
        total_comm_time_to_server += end - start

        start = time.perf_counter()
        print("[DEBUG] 모델 전송 완료, 평균 모델 수신 대기 중")
        avg_state_dict = recv_pickle(client_socket)
        print("[DEBUG] 평균 모델 수신 완료")
        end = time.perf_counter()
        print(f"서버 -> 클라이언트 전송 시간: {end - start:.4f}초")
        total_comm_time_from_server += end - start

        model.load_state_dict(avg_state_dict)
        print("평균 모델 적용 완료")

    except Exception as e:
        print(f"에러 발생: {e}")
        break

client_socket.close()

print("\n 전체 라운드 완료")
print(f"총 학습 시간: {total_train_time:.4f}s")
print(f"총 통신 지연 시간(클라이언트->서버): {total_comm_time_to_server:.4f}s")
print(f"총 통신 지연 시간(서버->클라이언트): {total_comm_time_from_server:.4f}s")