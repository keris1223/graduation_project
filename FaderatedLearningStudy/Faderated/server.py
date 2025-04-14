import socket
import pickle
import torch
from model import SimpleCNN
import time
import os

HOST = '0.0.0.0'
PORT = 5000
NUM_CLIENTS = 16
NUM_ROUNDS = 20
print("Hello")
def recv_model(conn):
    start = time.perf_counter()
    data = b""
    while True:
        packet = conn.recv(65536)
        if not packet:
            break
        data += packet
    end = time.perf_counter()
    print(f"모델 수신 완료, 지연 시간: {end - start:.4f}초")
    return pickle.loads(data)

def average_models(client_models):
    base_model = SimpleCNN()
    avg_state_dict = base_model.state_dict()

    for key in avg_state_dict:
        avg_state_dict[key] = sum(cm[key] for cm in client_models) / len(client_models)

    base_model.load_state_dict(avg_state_dict)
    return base_model

def send_model(conn, model):
    serialized = pickle.dumps(model.state_dict())
    conn.sendall(serialized)
    conn.shutdown(socket.SHUT_WR)

# 소켓 열기 (한 번만)
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT))
server_socket.listen(NUM_CLIENTS)
print(f"서버 시작됨: {HOST}:{PORT}")

# 라운드 반복
for round_num in range(1, NUM_ROUNDS + 1):
    print(f"\nRound {round_num}/{NUM_ROUNDS} 시작")

    client_models = []
    client_connections = []

    for i in range(NUM_CLIENTS):
        conn, addr = server_socket.accept()
        print(f"[{i+1}/{NUM_CLIENTS}] 클라이언트 연결 완료: {addr}")
        conn.settimeout(10)
        model_params = recv_model(conn)
        client_models.append(model_params)
        client_connections.append(conn)

    averaged_model = average_models(client_models)

    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = f"{model_dir}/round_{round_num:02d}.pt"
    torch.save(averaged_model.state_dict(), model_path)
    print(f"Round {round_num} 평균 모델 저장: {model_path}")

    print("평균 모델 클라이언트에 전송 중...")
    for i, conn in enumerate(client_connections):
        send_model(conn, averaged_model)
        conn.close()
        print(f"  ↳ 클라이언트 {i+1} 전송 완료")

print("\n🎉 연합 학습 완료")
server_socket.close()
