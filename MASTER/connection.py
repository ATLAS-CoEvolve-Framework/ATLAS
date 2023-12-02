from fastapi import WebSocket, WebSocketDisconnect
from typing import List
import json
import time
from datetime import datetime


master_path = 'D:\FILES\OFFICIAL\FINAL\DistributedCompute\MASTER'
# master_path = '/home/dsmaster/DistributedCompute/MASTER'


with open(f'{master_path}/config.json', 'r') as cnf:
    conf = json.load(cnf)


with open(f'{master_path}/container_cred.json') as secret:
    container_cred = json.load(secret)
    container_config = {'type': "container_cred",
                        "args": container_cred}


# NETWORK ARCH
with open(f"{master_path}/{conf['gen_name']}") as gen_js, open(f"{master_path}/{conf['disc_name']}") as disc_js:
    gen_json, disc_json = json.load(gen_js), json.load(disc_js)
    neural = {
        'type': 'config_neural',
        'args': {
            'generator': gen_json,
            'discriminator': disc_json
        }
    }

filenames = conf['files']

file_downloads = [{'type': "download_blob", "filename": x,
                   "redownload": "False"} for x in filenames]

calling = [{'type': 'call'}]
configs = calling#[container_config, neural] + file_downloads + calling


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.WORKERS = {}

    async def connect(self, websocket: WebSocket, client_id):
        await websocket.accept()
        self.active_connections.append(websocket)

        self.WORKERS[client_id] = {
            "conn": websocket, "status": "Connecting", 'ttl': time.time(), 'ip': websocket.client.host, 'bench':{}}

        current_time = datetime.now().strftime("%H:%M:%S")


        print(f"{client_id} connected at {websocket.client.host} - {current_time}")

        for config in configs:
            await websocket.send_text(json.dumps(config))

    def disconnect(self, client_id):
        try:
            socket = self.WORKERS[client_id]['conn']
            self.active_connections.remove(socket)
            self.WORKERS.pop(client_id, None)
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"Node {client_id} disconnected at {current_time}")
        except Exception as e:
            print(e)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)
