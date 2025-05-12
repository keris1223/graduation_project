import socket
import pickle
import torch
from model import BigCNN
import time
import os
import threading

HOST = '0.0.0.0'
PORT = 5000
NUM_CLIENTS = 8
NUM_ROUNDS = 10
print("Hello")
def recv_model_thread(conn, results, index):
    start = time.perf_counter()
    data = b""
    while True:
        packet = conn.recv(1048576)
        if not packet:
            break
        data += packet
    end = time.perf_counter()
    print(f"모델 수신 완료, 지연 시간: {end - start:.4f}초")
    model_state = pickle.loads(data)
    results[index] = model_state

def send_model_thread(conn,model):
    serialized = pickle.dumps(model.state_dict())
    conn.sendall(serialized)
    conn.shutdown(socket.SHUT_WR)
    conn.close()

def average_models(client_models):
    base_model = BigCNN()
    avg_state_dict = base_model.state_dict()

    sum_start = time.perf_counter()
    for key in avg_state_dict:
        avg_state_dict[key] = sum(cm[key] for cm in client_models) / len(client_models)
    sum_end = time.perf_counter()

    print(f"모델 평균 계산 완료, 시간: {sum_end-sum_start:.4f}s")

    base_model.load_state_dict(avg_state_dict)
    return base_model

# 소켓 열기 (한 번만)
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT))
server_socket.listen(NUM_CLIENTS)
print(f"서버 시작됨: {HOST}:{PORT}")

total_round_recv_time = 0.0
total_round_send_time = 0.0

# 라운드 반복
for round_num in range(1, NUM_ROUNDS + 1):
    print(f"\nRound {round_num}/{NUM_ROUNDS} 시작")

    client_connections = []

    for i in range(NUM_CLIENTS):
        conn, addr = server_socket.accept()
        print(f"[{i+1}/{NUM_CLIENTS}] 클라이언트 연결 완료: {addr}")
        client_connections.append(conn)

    recv_threads = []
    client_models = [None] * NUM_CLIENTS
    recv_start = time.perf_counter()

    for idx, conn in enumerate(client_connections):
        t = threading.Thread(target=recv_model_thread, args=(conn, client_models, idx))
        t.start()
        recv_threads.append(t)

    for t in recv_threads:
        t.join()

    recv_end = time.perf_counter()
    recv_time = recv_end - recv_start
    print(f"Round {round_num} 클라이언트로부터 수신 시간: {recv_time:.4f}초")
    total_round_recv_time += recv_time

    learn_start = time.perf_counter()
    averaged_model = average_models(client_models)
    learn_end = time.perf_counter()
    print(f"Round {round_num} 수신받은 파라미터의 평균 계산 시간: {learn_end - learn_start:.4f}초")


    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = f"{model_dir}/round_{round_num:02d}.pt"
    torch.save(averaged_model.state_dict(), model_path)
    print(f"Round {round_num} 평균 모델 저장: {model_path}")

    print("평균 모델 클라이언트에 전송 시작")
    total_send_start = time.perf_counter()
    send_threads = []

    for conn in client_connections:
        t = threading.Thread(target=send_model_thread, args=(conn, averaged_model))
        t.start()
        send_threads.append(t)
    for t in send_threads:
        t.join()

    total_send_end = time.perf_counter()
    total_send = total_send_end - total_send_start
    print(f"Round {round_num} 전송 시간: {total_send:.4f}s")

    total_round_send_time += total_send

print("\n연합 학습 완료")
print(f"전체 수신 시간 총합: {total_round_recv_time:.4f}초")
print(f"전체 전송 시간 총합: {total_round_send_time:.4f}초")
server_socket.close()
