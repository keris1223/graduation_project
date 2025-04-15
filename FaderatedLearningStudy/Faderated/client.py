import socket
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import BigCNN
import time

SERVER_IP = '14.47.39.5'  # 예: '192.168.0.101'
PORT = 5000
NUM_ROUNDS = 10

total_train_time = 0.0
total_comm_time_to_server = 0.0
total_comm_time_from_server = 0.0

def receive_model(sock):
    data = b""
    while True:
        packet = sock.recv(1048576)
        if not packet:
            break
        data += packet
    state_dict = pickle.loads(data)
    return state_dict


# 모델 및 학습
model = BigCNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')
model.to(device)

transform = transforms.ToTensor()
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transform),
    batch_size=32, shuffle=True
)

optimizer = optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

for round_num in range(1, NUM_ROUNDS + 1):
    print(f"\nRound {round_num}/{NUM_ROUNDS} 시작")

    train_start = time.perf_counter()

    # 로컬 학습 (한 배치만)
    model.train()
    for epoch in range(3):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            # if batch_ids >= 5:

    train_end = time.perf_counter()
    print(f"로컬 학습 시간: {train_end - train_start:.4f}s")
    total_train_time += train_end - train_start

    # 모델 파라미터 직렬화
    serialized_params = pickle.dumps(model.state_dict())

    # 서버에 전송
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((SERVER_IP, PORT))

        start = time.perf_counter()

        client_socket.sendall(serialized_params)
        client_socket.shutdown(socket.SHUT_WR)

        end = time.perf_counter()
        print(f"클라이언트->서버 전송 지연 시간: {end - start:.4f}초")
        total_comm_time_to_server += end - start

        start = time.perf_counter()
        avg_state_dict = receive_model(client_socket)
        end = time.perf_counter()
        print(f"서버 -> 클라이언트 (서버에서 모델을 모두 받아 합하여 클라로 전송) 시간: {end - start:.4f}초")
        total_comm_time_from_server += end - start

        model.load_state_dict(avg_state_dict)
        client_socket.close()

        print("평균 모델 적용 완료")

    except Exception as e:
        print(f"서버 연결 실패: {e}")
        break

print("\n 전체 라운드 완료")
print(f"총 학습 시간: {total_train_time:.4f}s")
print(f"총 통신 지연 시간(클라이언트->서버): {total_comm_time_to_server:.4f}s")
print(f"총 통신 지연 시간(서버->클라이언트): {total_comm_time_from_server:.4f}s")