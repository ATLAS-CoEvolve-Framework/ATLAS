#!/bin/env python3.9




import os
import shutil
from routine import Evolve, WAIT
from connection import ConnectionManager, download_files_parallel, master_path, conf, upload_file
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
import asyncio
import json
import time
import random
from fastapi.staticfiles import StaticFiles
import pickle
from hypercorn.config import Config
from hypercorn.asyncio import serve
import nest_asyncio
from threading import Thread
nest_asyncio.apply()
import ast
import logging




logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)


restore = False
script = []
remain = []
minimum_nodes = 1
counter = 0



app = FastAPI()
config = Config()
config.bind = ["0.0.0.0:10000"]
manager = ConnectionManager()


evolver = Evolve(conf['M'], conf['N'], conf['dataset'], epochs=conf['epochs'], features=conf['features'],
                 mutator=conf['mutator'], limits=conf['limits'], disc_metric=conf['disc_metric'], gen_metric=conf['gen_metric'],)


def msg_parser(data)->dict:
    try:
        if type(data) == dict:
            msg=data
        elif type(data) == str:
            try:
                msg = json.loads(data)
            except:
                msg=ast.literal_eval(data)  
    except ValueError:
        print("Could not parse:",data, " of type ", type(data))
        msg=None
    return msg


def remove_from(script: list[str], item):
    ret = [d for d in script if msg_parser(d).get('type') not in ["WAIT", "create"]]
    fin =  [d for d in ret if str(msg_parser(d).get('id')) != str(item)]
    return fin



def blob_snapshot(snap_name: str, evolver: Evolve, subspace=False):
    files =[]
    files+=[x+evolver.EXT for x in evolver.gans.keys()]

    if subspace:
        files+=[x+'_copy'+evolver.EXT for x in evolver.subspace_d.keys()]
        files+=[x+'_copy'+evolver.EXT for x in evolver.subspace_g.keys()]

    download_files_parallel(files, base_path=snap_name)
    shutil.make_archive(snap_name, 'zip', snap_name)
    upload_file(snap_name+'.zip')
    shutil.rmtree(snap_name)
    os.remove(snap_name+'.zip')




def logger():
    global evolver
    with open(f'{master_path}/log.json', 'a') as f1:
        existing_data = []
        try:
            with open(f'{master_path}/log.json', 'r') as f2:
                existing_data = json.load(f2)
        except FileNotFoundError:
            pass 

        new_data = {
            "evo_space": evolver.gans,
            "subspace_d": evolver.subspace_d,
            "subspace_g": evolver.subspace_g,
            "current_cell": evolver.curr_cell
        }
        existing_data.append(new_data)

        json.dump(existing_data, f1)  

    snap_name = f'snapshot_{counter}'
    print("Taking blob snapshot", snap_name)
    blob_snapshot(snap_name,evolver)


    # with open(f'{master_path}/log.txt', 'a') as f1:
    #     f1.write(json.dumps({"evo_space": evolver.gans, "subspace_d": evolver.subspace_d,
    #              "subspace_g": evolver.subspace_g, "current_cell": evolver.curr_cell}))
    #     f1.write('\n,\n')




def history(msg):
    with open(f'{master_path}/sent.txt', 'a') as f1:
        f1.write(msg)
        f1.write('\n,\n')




def watch_clients():
    while True:
        for client in list(manager.WORKERS.keys()):
            if (time.time() - manager.WORKERS[client]['ttl'] > 3):
                manager.disconnect(client)
        time.sleep(0.2)




def parse_msg(data, client_id):
    global remain


    if client_id in list(manager.WORKERS.keys()):
        manager.WORKERS[client_id]['ttl'] = time.time()  # heartbeat ping




    msg = msg_parser(data)


   


    try:
        manager.WORKERS[client_id]['bench'] = msg['bench']


        if (msg['type'] == 'status'):
            manager.WORKERS[client_id]['status'] = msg['payload']
    except KeyError as _:
        print("No benchmark found")


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: int):
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            parse_msg(data, client_id)


            # await manager.broadcast(f"Client #{client_id} says: {data}")


    except WebSocketDisconnect:
        manager.disconnect(client_id)
        await manager.broadcast(f"Client #{client_id} left the job")




@app.get("/stat")
async def stats():
    data = {"evo_space": evolver.gans, "subspace_d": evolver.subspace_d,
            "subspace_g": evolver.subspace_g, "current_cell": evolver.curr_cell,    "M": evolver.M,
            "N": evolver.N,
            "best": evolver.best}
    return data




@app.post("/update/{client_id}")
async def get_update(body: Request,  client_id: int):
    global remain
    msg = await body.json()


    try:


        if (msg['type'] == 'finished'):
            idx = msg['command']
            logging.info(f"Finished {idx}")
            remain = remove_from(remain, idx)
            manager.WORKERS[client_id]['status'] = "READY"


        elif (msg['type'] == 'fitness'):
            res:dict = msg['payload']
            payload:dict
            nn:str
            for nn, payload in res.items():
                for metric, value in payload.items():
                    if (nn.count('_copy')):
                        suffix = nn.removesuffix('_copy')
                        if (nn.startswith('gen')):
                            if suffix in list(evolver.subspace_g.keys()):
                                evolver.subspace_g[suffix]['fitness'][metric] = value
                        elif (nn.startswith('disc')):
                            if suffix in list(evolver.subspace_d.keys()):
                                evolver.subspace_d[suffix]['fitness'][metric] = value
                    else:
                        evolver.gans[nn]['fitness'][metric] = value




    except Exception as e:
        print(e)
        return {"error": e}


    return {"message": "Ack"}




@app.websocket("/stats")
async def websocket_endpoint_stat(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = {"evo_space": evolver.gans, "subspace_d": evolver.subspace_d,
                "subspace_g": evolver.subspace_g, "current_cell": evolver.curr_cell,
                "M": evolver.M,
                "N": evolver.N,
                "best": evolver.best if evolver.best is not None else "Waiting"}
        await websocket.send_json(data)
        await asyncio.sleep(1.5)




@app.websocket("/nodes")
async def node_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = {k: {'status': manager.WORKERS[k]['status'], 'ip': manager.WORKERS[k]['ip'],
                    'bench': manager.WORKERS[k]['bench']} for k in list(manager.WORKERS.keys())}
        await websocket.send_json(data)
        await asyncio.sleep(1.5)




@app.get("/var")
async def vars():
    global script, remain
    data = {"remain": remain, "Script": script}
    return data






async def compute_across_nodes():
    global script, remain


    command = ''
    wait = False
    time.sleep(2)


    while True:
        nodes = [x for x in list(manager.WORKERS.keys())
                 if manager.WORKERS[x]['status'] == 'READY']
        free_node = random.choice(nodes) if len(nodes) > 0 else None


        if len(nodes) == 0:
            time.sleep(5)
            continue


        if wait:
            waiting=time.time()
            time.sleep(3)
            wait = False
            while (len(nodes) != len(manager.WORKERS)):  # Wait till all connected nodes are idle
                nodes = [x for x in list(manager.WORKERS.keys()) if manager.WORKERS[x]['status'] == 'READY']
                wait = False
                if ((time.time() - waiting) > 300):
                    script = [x for x in script if x != WAIT]
                    break
                time.sleep(0.5)


            if (not wait and len(remain) > 0 and len(script) == 0):
                # if all nodes are idle and have stuff unfinished -> reassign incomplete jobs
                print("Reassigning Incomplete Jobs")
                script = remain.copy()
                print(remain)
                remain.clear()
                script.append(WAIT)


        # Wait for available nodes
        elif (free_node is None):
            time.sleep(5)
            continue


        # SEND COMMAND
        else:
            if (len(script) == 0):
                print("Part finished")
                return


            command = script.pop(0)
            if command == WAIT:
                wait = True
                waiting = time.time()
                continue


            # print(f"Sending {command} to Node {free_node}")
            time.sleep(0.5)
            remain.append(command)
            try:
                await manager.send_personal_message(str(command), manager.WORKERS[free_node]['conn'])
                # logging.info(f"{str(command)} - {manager.WORKERS[free_node]['ip']}")
            except Exception as e:
                print("Could not send command due to exception: ", e)
            history(str(command))
            nodes.remove(free_node)


        time.sleep(0.5)




def status_parser(nodes):
    old = None
    while True:
        op = [(x, nodes[x]['status']) for x in list(nodes.keys())]
        if (old != op):
            old = op
            print(op)
        time.sleep(1)




def async_caller():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)


    loop.run_until_complete(start_sequence())
    loop.close()




async def start_sequence():
    global script, remain, evolver
    # WAIT FOR CONNECTION
    while len(manager.WORKERS) < minimum_nodes:
        # print("Waiting for Nodes >", minimum_nodes)
        time.sleep(3)


    # BACKUP
    if (not restore):
        with open(f"{master_path}/evo.pkl", 'wb') as f1:  # Comment if restoring
            pickle.dump(evolver, f1)


    # RESTORE
    if (restore):
        with open(f"{master_path}/evo.pkl", 'rb') as f1:  # comment if starting new
            evolver = pickle.load(f1)


    script = evolver.script.copy()
    await compute_across_nodes()  # send off to compute


    logger()


    for _ in range(3*conf['generations']*conf['M']*conf['N']):


        print("Subspace of cell ", evolver.curr_cell)


        evolver.next()


        with open(f"{master_path}/evo.pkl", 'wb') as f1:
            pickle.dump(evolver, f1)


        script = evolver.script.copy()
        await compute_across_nodes()


        logger()




if __name__ == '__main__':


    starter = Thread(target=async_caller, args=())
    starter.start()


    watcher = Thread(target=watch_clients, args=())
    watcher.start()


    asyncio.run(serve(app, config))