import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import socket
import pickle
import torch
import threading
import os

from utils import send_pickle, recv_pickle
import time
import numpy as np
from llm_model import load_model
from peft import set_peft_model_state_dict, get_peft_model_state_dict

HOST = '0.0.0.0'
PORT = 5000
NUM_CLIENTS = 2
NUM_ROUNDS = 100

final_ack_barrier = threading.Barrier(NUM_CLIENTS)

client_connections = []
client_models = [None] * NUM_CLIENTS
client_losses = [None] * NUM_CLIENTS
client_epochs = [3] * NUM_CLIENTS  # 초기값 3
model_ready_barrier = threading.Barrier(NUM_CLIENTS)
send_ready_barrier = threading.Barrier(NUM_CLIENTS)
averaged_model = None
comm_times = [0.0] * NUM_CLIENTS

averaged_model = load_model()

client_log = open("client_log.csv", "w")
client_log.write("round,client_id,loss,comm_time,epoch\n")

round_log = open("round_time.csv", "w")
round_log.write("round,total_time_sec\n")


def average_models(models):
    avg_state = {}
    for key in models[0].keys():
        avg_state[key] = sum(m[key] for m in models) / len(models)
    return avg_state


def decide_epochs_from_losses(losses, min_epoch=3, max_epoch=7):
    median = np.median(losses)
    std = np.std(losses) + 1e-6

    epochs = []
    for loss in losses:
        z = (loss - median) / std
        z = np.clip(z, -2, 2)
        norm = 0.5 + 0.25 * z
        epoch = int(round(min_epoch + norm * (max_epoch - min_epoch)))
        epochs.append(epoch)

    return epochs


def handle_client(conn, client_id):
    global averaged_model

    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"[Client {client_id}] Round {round_num} 수신 대기 중...")
        try:
            recv_start = time.perf_counter()
            data = recv_pickle(conn)
            recv_end = time.perf_counter()
            comm_times[client_id] = recv_end - recv_start

            client_models[client_id] = data['model']
            client_losses[client_id] = data['metric']['loss']
            print(f"[Client {client_id}] 수신 완료 - Loss: {client_losses[client_id]:.4f}, CommTime: {comm_times[client_id]:.4f}s")
        except Exception as e:
            print(f"[Client {client_id}] 수신 실패: {e}")
            break
        model_ready_barrier.wait()

        # 서버 쓰레드 하나만 평균 계산
        if client_id == 0:
            print(f"[Server] Round {round_num} 모델 평균 계산 중...")
            avg_start = time.perf_counter()
            avg_state = average_models(client_models)
            set_peft_model_state_dict(averaged_model, avg_state)
            avg_end = time.perf_counter()
            print(f"[Server] 평균 계산 소요 시간: {avg_end - avg_start:.4f}s")

            torch.save(averaged_model.state_dict(), f"models/round_{round_num:02d}.pt")
            print(f"[Server] 평균 모델 저장 완료")

            # 다음 라운드 학습 에폭 계산
            epochs = decide_epochs_from_losses(client_losses)
            for i in range(NUM_CLIENTS):
                
                client_epochs[i] = epochs[i]
                print(f"[Server] Client {i} → Epoch 설정: {client_epochs[i]} (Loss: {client_losses[i]:.4f})")

        send_ready_barrier.wait()
        loss = client_losses[client_id]
        comm_time = comm_times[client_id]
        epoch = client_epochs[client_id]
        client_log.write(f"{round_num},{client_id},{loss:.4f},{comm_time:.4f},{epoch}\n")
        client_log.flush()
        # 평균 모델 + 다음 에폭 수 전송
        try:
            if round_num < NUM_ROUNDS:
                payload = {
                    'model': get_peft_model_state_dict(averaged_model),
                    'local_epochs': client_epochs[client_id]
                }
                send_start = time.perf_counter()
                send_pickle(conn, payload)
                send_end = time.perf_counter()
                print(f"[Client {client_id}] 평균 모델 및 Epoch 전송 완료 - 전송 시간: {send_end - send_start:.4f}s")
        except Exception as e:
            print(f"[Client {client_id}] 전송 실패: {e}")
            break

        if round_num == NUM_ROUNDS:
            try:
                done_signal = recv_pickle(conn)
                if done_signal.get("done") is True:
                    print(f"[Client {client_id}] done 메시지 수신 완료")

                    # 모든 클라이언트가 done 메시지 보낼 때까지 대기
                    final_ack_barrier.wait()
            except Exception as e:
                print(f"[Client {client_id}] done 수신 또는 ACK 전송 실패: {e}")
    conn.close()
    print(f"[Client {client_id}] 연결 종료됨")


# 서버 소켓 수신 대기
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT))
server_socket.listen(NUM_CLIENTS)
print(f"[Server] Listening on {HOST}:{PORT}")

os.makedirs("models", exist_ok=True)

threads= []
for i in range(NUM_CLIENTS):
    conn, addr = server_socket.accept()
    print(f"[Server] Client {i} connected from {addr}")
    client_connections.append(conn)
    t = threading.Thread(target=handle_client, args=(conn, i))
    t.start()
    threads.append(t)

for t in threads:
    t.join()

client_log.close()
round_log.close()
print("[Server] 로그 파일 닫힘")