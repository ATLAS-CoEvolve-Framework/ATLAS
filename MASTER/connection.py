from fastapi import WebSocket
from typing import List
import json
import time
from datetime import datetime
import os
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential


master_path = 'D:\FILES\OFFICIAL\FINAL\DistributedCompute\MASTER'
# master_path = '/home/dsmaster/DistributedCompute/MASTER'


with open(f'{master_path}/config.json', 'r') as cnf:
    conf = json.load(cnf)


with open(f'{master_path}/container_cred.json') as secret:
    container_cred = json.load(secret)
    container_config = {'type': "container_cred",
                        "args": container_cred}

default_credential = DefaultAzureCredential()
connect_str = container_cred['connect_str']
container_name = container_cred['container_name']
blob_service_client = BlobServiceClient.from_connection_string(connect_str)
container_client = blob_service_client.get_container_client(container=container_name)



import time, os, threading
def upload_file(path):
    print(f"Starting upload of {path}")
    begin = time.time()
    blob_client = blob_service_client.get_blob_client(
        container=container_name, blob=path)
    with open(path, 'rb') as file:
        blob_client.upload_blob(file, overwrite=True, connection_timeout=600)
    print(f"Upload Complete {path} in {(time.time()-begin):.2f}s")


def download_file(path, redownload=True, base_path = ''):
    if (not redownload and os.path.exists(path)):
        print('File Exists')
        return
    print(f"Starting download of {path}")
    begin = time.time()
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    with open(os.path.join(base_path,path), "wb") as download_file:
        download_file.write(container_client.download_blob(path).readall())
        print(f"Download of {path} complete in {(time.time()-begin):.2f}s")


def download_files_parallel(paths: list, redownload=True, base_path=''):
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    threads = []
    for path in paths:
        thread = threading.Thread(target=download_file, args=(path, redownload, base_path))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()




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
