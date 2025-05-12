import socket
import pickle
import torch
from model import BigCNN
import threading
import os
import time

HOST = '0.0.0.0'
PORT = 5000
NUM_CLIENTS = 8
NUM_ROUNDS = 10

client_connections = []
client_models = [None] * NUM_CLIENTS
model_ready_barrier = threading.Barrier(NUM_CLIENTS)
send_ready_barrier = threading.Barrier(NUM_CLIENTS)
averaged_model = None
lock = threading.Lock()

def average_models(models):
    base_model = BigCNN()
    avg_state_dict = base_model.state_dict()
    for key in avg_state_dict:
        avg_state_dict[key] = sum(model[key] for model in models) / len(models)
    base_model.load_state_dict(avg_state_dict)
    return base_model

def handle_client(conn, client_id):
    global averaged_model

    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"[Client {client_id}] Round {round_num}: Receiving model...")
        data = b""
        while True:
            packet = conn.recv(1048576)
            if not packet:
                break
            data += packet

        model_state = pickle.loads(data)
        client_models[client_id] = model_state
        print(f"[Client {client_id}] Model received.")

        model_ready_barrier.wait()  # 모든 클라이언트 수신 대기

        if client_id == 0:
            print(f"[Server] Averaging models for Round {round_num}...")
            start = time.perf_counter()
            averaged_model = average_models(client_models)
            end = time.perf_counter()
            print(f"[Server] Model averaging completed in {end - start:.4f}s")

            # 저장
            model_dir = "models"
            os.makedirs(model_dir, exist_ok=True)
            torch.save(averaged_model.state_dict(), f"{model_dir}/round_{round_num:02d}.pt")
            print(f"[Server] Saved averaged model: round_{round_num:02d}.pt")

        send_ready_barrier.wait()  # 모델 계산 완료 대기

        serialized = pickle.dumps(averaged_model.state_dict())
        conn.sendall(serialized)
        print(f"[Client {client_id}] Averaged model sent.")

    conn.close()
    print(f"[Client {client_id}] Disconnected.")

# 소켓 설정
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT))
server_socket.listen(NUM_CLIENTS)
print(f"[Server] Listening on {HOST}:{PORT}")

# 클라이언트 연결 수락 및 스레드 시작
for i in range(NUM_CLIENTS):
    conn, addr = server_socket.accept()
    print(f"[Server] Client {i} connected from {addr}")
    client_connections.append(conn)
    threading.Thread(target=handle_client, args=(conn, i)).start()
