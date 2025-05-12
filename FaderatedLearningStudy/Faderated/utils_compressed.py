import pickle
import struct
import gzip
import io

def send_pickle(sock, obj):
    """gzip 압축 후 pickle 데이터 전송"""
    raw = pickle.dumps(obj)
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode='wb') as gz:
        gz.write(raw)
    compressed = buffer.getvalue()
    length = struct.pack('>I', len(compressed))
    sock.sendall(length)
    sock.sendall(compressed)
    print(f"[DEBUG] send_pickle: 원본 {len(raw)} → 압축 {len(compressed)} bytes 전송")

def recv_pickle(sock):
    """gzip 압축된 pickle 데이터 수신 및 해제"""
    length_data = sock.recv(4)
    if not length_data or len(length_data) < 4:
        raise ConnectionError("길이 헤더 수신 실패")
    total_len = struct.unpack('>I', length_data)[0]
    print(f"[DEBUG] recv_pickle: 압축 데이터 {total_len} bytes 수신 시작")

    data = b""
    while len(data) < total_len:
        packet = sock.recv(total_len - len(data))
        if not packet:
            raise ConnectionError("압축 데이터 수신 도중 연결 종료")
        data += packet
        print(f"[DEBUG] 수신 중: {len(data)}/{total_len} bytes")

    buffer = io.BytesIO(data)
    with gzip.GzipFile(fileobj=buffer, mode='rb') as gz:
        raw = gz.read()
    print(f"[DEBUG] 압축 해제 후 크기: {len(raw)} bytes")
    return pickle.loads(raw)
