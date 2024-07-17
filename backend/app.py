import asyncio
import json
import os
import threading
import importlib
from pathlib import Path
from queue import Empty, Queue
from threading import Thread, Event
from typing import Dict, List, Optional

import aiofiles
import uvicorn
import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from loguru import logger

from backend.utils.connection import manager
from simulation.helpers.message import message_manager, MessageUnit
from simulation.helpers.events import play_event, stop_event
from backend.utils.body_models import (
    ModelConfigs,
    MemoryConfig,
    CheckpointReq,
    CheckpointResp,
    FilterCondition,
)


app = FastAPI()
proj_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
templates = Jinja2Templates(directory=os.path.join(proj_path, "backend", "templates"))


simulation_thread: Thread
events: Dict[str, Event] = {}
queue = Queue()
simulator = None
lock = threading.Lock()


# async def fetch_msg(websocket: WebSocket):
#     try:
#         while True:
#             # wait for new message and send to frontend via WebSocket
#             if not message_queue.empty():
#                 msg = message_queue.get()
#                 if isinstance(msg, MessageUnit):
#                     await websocket.send_text(msg.model_dump_json())
#                 elif isinstance(msg, str):
#                     await websocket.send_text(msg)
#             await asyncio.sleep(1)  # polling frequency
#     except WebSocketDisconnect:
#         print("WebSocket connection closed")


def check_msg_filter(
    msg: MessageUnit, filter_condition: Optional[FilterCondition]
) -> bool:
    if filter_condition is None:
        return True
    elif filter_condition.type == "turn":
        return msg.turn in filter_condition["turns"]
    elif filter_condition.type == "id":
        return msg.agent_id in filter_condition["ids"]
    elif filter_condition.type == "name":
        return msg.name.lower() in [name.lower() for name in filter_condition["names"]]
    return False


def filter_messages(
    msgs: List[MessageUnit], filter_condition: Optional[FilterCondition]
) -> List[MessageUnit]:
    return [msg for msg in msgs if check_msg_filter(msg, filter_condition)]


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    filter_condition = None

    async def receive_filter_condition(websocket: WebSocket):
        data = await websocket.receive_text()
        return json.loads(data)

    async def send_filtered_history(filter_condition: Optional[FilterCondition]):
        with lock:
            msgs = message_manager.messages.copy()
        msgs = filter_messages(msgs, filter_condition)
        for msg in msgs:
            await manager.send(msg.model_dump_json())

    try:
        while True:
            # get msg from message queue
            try:
                msg = message_manager.message_queue.get_nowait()
            except Empty:
                msg = None

            # listen for filter condition
            filter_condition_task = asyncio.create_task(
                receive_filter_condition(websocket)
            )
            done, pending = await asyncio.wait([filter_condition_task], timeout=0.01)

            if filter_condition_task in done:
                filter_condition = filter_condition_task.result()
                logger.info(f"Receive new filter condition: {filter_condition}")
                filter_condition = FilterCondition(**filter_condition)
                if filter_condition.type == "None":
                    filter_condition = None
                await send_filtered_history(filter_condition)

            # send msg to frontend
            if msg and (
                not filter_condition or check_msg_filter(msg, filter_condition)
            ):
                await manager.send(msg.model_dump_json())
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    finally:
        manager.disconnect(websocket)


@app.websocket("/ws/{id}")
async def websocket_agent_endpoint(websocket: WebSocket, id: int):
    await manager.connect(websocket, id)
    try:
        while True:
            data = await websocket.receive_text()
            # TODO
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    finally:
        manager.disconnect(websocket)


@app.get("/agents")
def get_agents(query: Optional[str] = None):
    agents = simulator.agents

    def fuzzy_search(agents, query):
        return [agent for agent in agents if query.lower() in agent.name.lower()]

    if query:
        agents = fuzzy_search(agents, query)
    return agents


@app.get("/agent/{id}")
def get_agent(id: str):
    agents = simulator.agents
    try:
        return agents[id]
    except IndexError:
        return HTMLResponse(content="Agent not found.", status_code=404)


@app.put("/agent/{id}")
def put_agent(id: str, new_agent):
    agents = simulator.agents
    try:
        agent = agents[id]
        agent.update_from_dict(new_agent)
        return {"status": "success"}
    except IndexError:
        return HTMLResponse(content="Agent not found.", status_code=404)


@app.get("/model/{scene}", response_model=ModelConfigs)
async def get_model_configs(scene: str = "job_seeking"):
    config_file = os.path.join(
        proj_path, "simulation", "examples", scene, "configs", "model_configs.json"
    )
    logger.info(f"Get model config from {config_file}")
    async with aiofiles.open(config_file, "r") as f:
        model_configs = await f.read()
        model_configs = json.loads(model_configs)
        return model_configs


@app.put("/model/{scene}", response_model=ModelConfigs)
async def put_model_configs(model_configs: ModelConfigs, scene: str = "job_seeking"):
    config_file = os.path.join(
        proj_path, "simulation", "examples", scene, "configs", "model_configs.json"
    )
    logger.info(f"Put model config to {config_file}")
    async with aiofiles.open(config_file, "w") as f:
        await f.write(model_configs)
    return {"status": "success"}


@app.get("/memory/{scene}", response_model=MemoryConfig)
async def get_memory_config(scene: str = "job_seeking"):
    config_file = os.path.join(
        proj_path, "simulation", "examples", scene, "configs", "memory_configs.json"
    )
    logger.info(f"Get memory config from {config_file}")
    async with aiofiles.open(config_file, "r") as f:
        memory_configs = await f.read()
        memory_configs = json.loads(memory_configs)
        return memory_configs


@app.put("/memory/{scene}", response_model=MemoryConfig)
async def put_memory_config(memory_config: MemoryConfig, scene: str = "job_seeking"):
    config_file = os.path.join(
        proj_path, "simulation", "examples", scene, "configs", "memory_configs.json"
    )
    logger.info(f"Put memory config to {config_file}")
    async with aiofiles.open(config_file, "w") as f:
        await f.write(memory_config)
    return {"status": "success"}


@app.get("/checkpoint/{scene}", response_model=List[CheckpointResp])
def get_checkpoint(scene: str = "job_seeking"):
    runs = Path(os.path.join(proj_path, "simulation", "examples", scene, "runs"))
    if not runs.exists():
        runs.mkdir()
    checkpoints = runs.glob("*/*.pkl")
    run_names = [checkpoint.parent for checkpoint in checkpoints]
    resp = []
    for run_name in run_names:
        checkpoints = run_name.glob("*.pkl")
        resp.append(
            CheckpointResp(
                run_name=run_name.name,
                pkls=[checkpoint.name for checkpoint in checkpoints],
            )
        )
    return resp


@app.post("/checkpoint/{scene}", response_model=CheckpointReq)
def load_checkpoint(checkpoint_req: CheckpointReq, scene: str = "job_seeking"):
    logger.info(f"Load checkpoint from {checkpoint_req.run_name}/{checkpoint_req.pkl}")
    checkpoint_path = os.path.join(
        proj_path,
        "simulation",
        "examples",
        scene,
        "runs",
        checkpoint_req.run_name,
        checkpoint_req.pkl,
    )
    simulation_config_path = os.path.join(
        proj_path, "simulation", "examples", scene, "configs", "simulation_config.yml"
    )
    with open(simulation_config_path, "r") as f:
        simulation_config = yaml.safe_load(f)
    simulation_config["restore_file_path"] = checkpoint_path
    with open(simulation_config_path, "w") as f:
        yaml.safe_dump(simulation_config, f)
    return {"status": "success"}


@app.get("/messages", response_model=List[MessageUnit])
def get_messages(filter_condition: Optional[FilterCondition] = None):
    with lock:
        msgs = message_manager.messages.copy()
    return filter_messages(msgs, filter_condition)


@app.post("/start/{scene}")
async def start(scene: str = "job_seeking"):
    module_path = f"simulation.examples.{scene}.simulator"
    Simulator = importlib.import_module(module_path).Simulator
    global simulator
    simulator = Simulator()
    simulation_thread = Thread(target=simulator.run)
    simulation_thread.start()
    play_event.set()
    return {"status": "success"}


@app.post("/pause")
def pause_and_resume():
    if play_event.is_set():
        message_manager.message_queue.put("Pause simulation.")
        play_event.clear()
    else:
        message_manager.message_queue.put("Resume simulation.")
        play_event.set()
    return {"status": "success"}


@app.post("/stop")
async def stop():
    stop_event.set()
    return {"status": "success"}


@app.get("/", response_class=HTMLResponse)
async def read_root():
    return templates.TemplateResponse("index.html", {"request": {}})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
