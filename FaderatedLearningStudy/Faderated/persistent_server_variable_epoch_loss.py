import socket
import pickle
import torch
import threading
import os
from model import BigCNN
from utils import send_pickle, recv_pickle
import time

HOST = '0.0.0.0'
PORT = 5000
NUM_CLIENTS = 8
NUM_ROUNDS = 10

final_ack_barrier = threading.Barrier(NUM_CLIENTS)

client_connections = []
client_models = [None] * NUM_CLIENTS
client_losses = [None] * NUM_CLIENTS
client_epochs = [3] * NUM_CLIENTS  # 초기값 3
model_ready_barrier = threading.Barrier(NUM_CLIENTS)
send_ready_barrier = threading.Barrier(NUM_CLIENTS)
averaged_model = None


def average_models(models):
    base_model = BigCNN()
    avg_state = base_model.state_dict()
    for key in avg_state:
        avg_state[key] = sum(m[key] for m in models) / len(models)
    base_model.load_state_dict(avg_state)
    return base_model


def decide_epoch_from_loss(loss):
    if loss < 0.08:
        return 1
    elif loss < 0.4:
        return 2
    else:
        return 3


def handle_client(conn, client_id):
    global averaged_model

    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"[Client {client_id}] Round {round_num} 수신 대기 중...")
        try:
            data = recv_pickle(conn)
            client_models[client_id] = data['model']
            client_losses[client_id] = data['metric']['loss']
            print(f"[Client {client_id}] 수신 완료 - Loss: {client_losses[client_id]:.4f}")
        except Exception as e:
            print(f"[Client {client_id}] 수신 실패: {e}")
            break

        model_ready_barrier.wait()

        # 서버 쓰레드 하나만 평균 계산
        if client_id == 0:
            print(f"[Server] Round {round_num} 모델 평균 계산 중...")
            averaged_model = average_models(client_models)
            torch.save(averaged_model.state_dict(), f"models/round_{round_num:02d}.pt")
            print(f"[Server] 평균 모델 저장 완료")

            # 다음 라운드 학습 에폭 계산
            for i in range(NUM_CLIENTS):
                loss = client_losses[i]
                client_epochs[i] = decide_epoch_from_loss(loss)
                print(f"[Server] Client {i} → Epoch 설정: {client_epochs[i]} (Loss: {loss:.4f})")

        send_ready_barrier.wait()

        # 평균 모델 + 다음 에폭 수 전송
        try:
            if round_num < NUM_ROUNDS:
                payload = {
                    'model': averaged_model.state_dict(),
                    'local_epochs': client_epochs[client_id]
                }
                send_pickle(conn, payload)
                print(f"[Client {client_id}] 평균 모델 및 Epoch 전송 완료")
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

for i in range(NUM_CLIENTS):
    conn, addr = server_socket.accept()
    print(f"[Server] Client {i} connected from {addr}")
    client_connections.append(conn)
    threading.Thread(target=handle_client, args=(conn, i)).start()
