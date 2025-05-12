import socket
import pickle
import torch
from model import BigCNN
import threading
import os
import time
from utils import recv_pickle, send_pickle

HOST = '0.0.0.0'
PORT = 5000
NUM_CLIENTS = 8
NUM_ROUNDS = 10

client_connections = []
client_models = [None] * NUM_CLIENTS
model_ready_barrier = threading.Barrier(NUM_CLIENTS)
send_ready_barrier = threading.Barrier(NUM_CLIENTS)
averaged_model = None

def average_models(models):
    base_model = BigCNN()
    avg_state_dict = base_model.state_dict()
    for key in avg_state_dict:
        avg_state_dict[key] = sum(m[key] for m in models) / len(models)
    base_model.load_state_dict(avg_state_dict)
    return base_model

def handle_client(conn, client_id):
    global averaged_model

    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"[Client {client_id}] Round {round_num}: Receiving model...")
        try:
            model_state = recv_pickle(conn)
            if model_state is None:
                print(f"[Client {client_id}] 수신 실패 또는 연결 종료")
                break
            client_models[client_id] = model_state
        except Exception as e:
            print(f"[Client {client_id}] 수신 중 오류: {e}")
            break

        print(f"[Client {client_id}] Round {round_num}: 모델 수신 완료, barrier 진입 전")
        model_ready_barrier.wait()
        print(f"[Client {client_id}] Round {round_num}: barrier 통과")

        if client_id == 0:
            print(f"[Server] Round {round_num} 평균 계산 중...")
            averaged_model = average_models(client_models)
            torch.save(averaged_model.state_dict(), f"models/round_{round_num:02d}.pt")
            print(f"[Server] Round {round_num} 모델 저장 완료")

        send_ready_barrier.wait()

        try:
            send_pickle(conn, averaged_model.state_dict())
            print(f"[Client {client_id}] Round {round_num}: 평균 모델 전송 완료")
        except Exception as e:
            print(f"[Client {client_id}] 전송 중 오류: {e}")
            break

    conn.close()
    print(f"[Client {client_id}] 연결 종료됨")

# 소켓 수신
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT))
server_socket.listen(NUM_CLIENTS)
print(f"[Server] Listening on {HOST}:{PORT}")

for i in range(NUM_CLIENTS):
    conn, addr = server_socket.accept()
    print(f"[Server] Client {i} connected from {addr}")
    client_connections.append(conn)
    threading.Thread(target=handle_client, args=(conn, i)).start()
