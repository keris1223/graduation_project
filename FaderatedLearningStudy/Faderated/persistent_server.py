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

total_recv_time = 0.0
total_avg_time = 0.0
total_send_time = 0.0
def average_models(models):
    base_model = BigCNN()
    avg_state_dict = base_model.state_dict()
    for key in avg_state_dict:
        avg_state_dict[key] = sum(m[key] for m in models) / len(models)
    base_model.load_state_dict(avg_state_dict)
    return base_model

def handle_client(conn, client_id):
    global averaged_model, total_recv_time, total_avg_time, total_send_time

    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"[Client {client_id}] Round {round_num}: Receiving model...")
        try:
            recv_start = time.perf_counter()
            model_state = recv_pickle(conn)
            recv_end = time.perf_counter()
            if model_state is None:
                print(f"[Client {client_id}] 수신 실패 또는 연결 종료")
                break

            client_models[client_id] = model_state
            elapsed = recv_end - recv_start
            print(f"[Client {client_id}] 모델 수신 완료: {elapsed:.4f}s")
            if client_id == 0:
                total_recv_time += elapsed

        except Exception as e:
            print(f"[Client {client_id}] 수신 중 오류: {e}")
            break

        model_ready_barrier.wait()

        if client_id == 0:
            print(f"[Server] Round {round_num} 평균 계산 중...")
            avg_start = time.perf_counter()
            averaged_model = average_models(client_models)
            avg_end = time.perf_counter()
            avg_time = avg_end - avg_start
            total_avg_time += avg_time
            print(f"[Server] 평균 완료: {avg_time:.4f}s")
            torch.save(averaged_model.state_dict(), f"models/round_{round_num:02d}.pt")
            print(f"[Server] Round {round_num} 모델 저장 완료")

        send_ready_barrier.wait()

        try:
            send_start = time.perf_counter()
            send_pickle(conn, averaged_model.state_dict())
            send_end = time.perf_counter()
            send_time = send_end - send_start
            print(f"[Client {client_id}] Round {round_num}: 평균 모델 전송 완료({send_time:.4f}s)")
            if client_id == 0:
                total_send_time += send_time
        except Exception as e:
            print(f"[Client {client_id}] 전송 중 오류: {e}")
            break

    conn.close()
    print(f"[Client {client_id}] 연결 종료됨")

    if client_id == 0:
        print("\n[Server] 연합 학습 완료")
        print(f"총 수신 시간: {total_recv_time:.4f}초")
        print(f"총 평균 계산 시간: {total_avg_time:.4f}초")
        print(f"총 전송 시간: {total_send_time:.4f}초")

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
