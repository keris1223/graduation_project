import struct
import pickle
import gzip
import io

def send_pickle(sock, obj):
    data = pickle.dumps(obj)
    length = struct.pack('>I', len(data))  # 4바이트 길이 정보 (big-endian)
    sock.sendall(length)
    sock.sendall(data)

def recv_pickle(sock):
    length_data = sock.recv(4)
    if not length_data or len(length_data) < 4:
        raise ConnectionError("Failed to receive length header")
    total_len = struct.unpack('>I', length_data)[0]
    print(f"[DEBUG] Expecting {total_len} bytes")

    data = b""
    while len(data) < total_len:
        remaining = total_len - len(data)
        packet = sock.recv(remaining)
        if not packet:
            raise ConnectionError("Connection closed during data recv")
        data += packet
        print(f"[DEBUG] Received {len(data)}/{total_len} bytes")
    return pickle.loads(data)

