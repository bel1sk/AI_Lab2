import urllib.request
import urllib.parse
import json
import socket
import base64
import time
import os
import subprocess

def capture():
    # 1. Start Chrome with remote debugging
    chrome_cmd = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        "--headless=new",
        "--remote-debugging-port=9222",
        "--window-size=1920,1080",
        "http://localhost:6006/#scalars"
    ]
    print("Launching Chrome:", " ".join(chrome_cmd))
    proc = subprocess.Popen(chrome_cmd)
    
    try:
        # Wait for Chrome CDP port to become available
        targets = None
        for attempt in range(15):
            time.sleep(1)
            try:
                req = urllib.request.urlopen("http://127.0.0.1:9222/json", timeout=2)
                targets = json.loads(req.read().decode('utf-8'))
                if targets:
                    break
            except Exception as e:
                pass
                
        if not targets:
            print("Failed to connect to Chrome debugging port!")
            return False

        tb_target = None
        for t in targets:
            if t.get('type') == 'page' and '6006' in t.get('url', ''):
                tb_target = t
                break
        
        if not tb_target:
            print("Could not find TensorBoard page target! Targets found:")
            for t in targets:
                print(t.get('type'), t.get('url'))
            return False

        ws_url = tb_target['webSocketDebuggerUrl']
        print("Found TensorBoard target:", tb_target.get('title'), ws_url)
        
        parsed = urllib.parse.urlparse(ws_url)
        host = parsed.hostname
        port = parsed.port or 9222
        path = parsed.path
        
        s = socket.create_connection((host, port))
        key = base64.b64encode(b"0123456789abcdef").decode()
        handshake = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            f"Upgrade: websocket\r\n"
            f"Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            f"Sec-WebSocket-Version: 13\r\n\r\n"
        )
        s.sendall(handshake.encode())
        
        res = b""
        while b"\r\n\r\n" not in res:
            chunk = s.recv(4096)
            if not chunk: break
            res += chunk
            
        assert b"101 " in res, f"Handshake failed: {res.decode('latin1', errors='ignore')}"
        print("Handshake successful!")
        
        def send_cmd(cmd_id, method, params=None):
            msg = {"id": cmd_id, "method": method}
            if params: msg["params"] = params
            payload = json.dumps(msg).encode('utf-8')
            length = len(payload)
            hdr = bytearray([0x81])
            mask_key = b"\x12\x34\x56\x78"
            if length <= 125:
                hdr.append(0x80 | length)
            elif length <= 65535:
                hdr.append(0x80 | 126)
                hdr.extend(length.to_bytes(2, 'big'))
            else:
                hdr.append(0x80 | 127)
                hdr.extend(length.to_bytes(8, 'big'))
            masked = bytearray(b ^ mask_key[i % 4] for i, b in enumerate(payload))
            s.sendall(hdr + mask_key + masked)

        def recv_msg():
            hdr = s.recv(2)
            while len(hdr) < 2:
                m = s.recv(2 - len(hdr))
                if not m: return None
                hdr += m
            length = hdr[1] & 0x7F
            if length == 126:
                lb = s.recv(2)
                while len(lb) < 2: lb += s.recv(2 - len(lb))
                length = int.from_bytes(lb, 'big')
            elif length == 127:
                lb = s.recv(8)
                while len(lb) < 8: lb += s.recv(8 - len(lb))
                length = int.from_bytes(lb, 'big')
            
            payload = bytearray()
            while len(payload) < length:
                chunk = s.recv(min(length - len(payload), 65536))
                if not chunk: break
                payload.extend(chunk)
            return json.loads(payload.decode('utf-8', errors='ignore'))

        # Wait 6 seconds for TensorBoard scalars & Polymer charts to load
        print("Waiting 6 seconds for charts to render...")
        time.sleep(6)
        
        # Request screenshot
        print("Requesting Page.captureScreenshot...")
        send_cmd(100, "Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False})
        
        while True:
            resp = recv_msg()
            if not resp:
                print("Connection closed without response")
                break
            if resp.get('id') == 100:
                data = resp.get('result', {}).get('data')
                if data:
                    img_bytes = base64.b64decode(data)
                    out_path1 = r"D:\Stud\AI\Lab2\tensorboard_screenshot.png"
                    out_path2 = r"D:\Stud\AI\Lab2\images\tensorboard_scalars.png"
                    os.makedirs(os.path.dirname(out_path2), exist_ok=True)
                    with open(out_path1, "wb") as f:
                        f.write(img_bytes)
                    with open(out_path2, "wb") as f:
                        f.write(img_bytes)
                    print(f"Successfully captured TensorBoard screenshot: {len(img_bytes)} bytes!")
                    print(f"Saved to: {out_path1} and {out_path2}")
                    return True
                else:
                    print("Error in response:", resp)
                    return False
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            pass
    return False

if __name__ == "__main__":
    capture()
