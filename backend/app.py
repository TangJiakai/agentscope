import asyncio
import json
import os
import random
import importlib
import inspect
from pathlib import Path
from queue import Empty, Queue
import subprocess
from threading import Thread, Event
from typing import Dict, List, Optional

import aiofiles
import uvicorn
import yaml
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from loguru import logger

from backend.utils.connection import manager
from simulation.helpers.message import message_manager, MessageUnit
from simulation.helpers.events import play_event, stop_event
from backend.utils.body_models import (
    Scene,
    ModelConfig,
    AgentConfig,
    MemoryConfig,
    PathReq,
    FilterCondition,
    DistributedArgs,
    DistributedConfig,
)
from backend.utils.utils import try_serialize_dict
from simulation.memory import (
    NoneMemory,
    ShortMemory,
    ShortLongMemory,
    ShortLongReflectionMemory,
)


proj_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
templates = Jinja2Templates(directory=os.path.join(proj_path, "backend", "templates"))


_scene = "job_seeking"
simulation_thread: Thread
events: Dict[str, Event] = {}
queue = Queue()
simulator = None
lock = asyncio.Lock()
distributed: bool = False
distributed_args: DistributedArgs = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # clean distributed servers
    if distributed:
        kill_server_sh_path = os.path.join(
            proj_path, "simulation", "examples", _scene, "kill_all_server.sh"
        )
        command = ["bash", kill_server_sh_path]
        try:
            result = subprocess.run(
                command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            logger.info("Kill server script output:\n", result.stdout.decode())
        except subprocess.CalledProcessError as e:
            logger.error("Kill server script failed with return code:", e.returncode)
            logger.error("Kill server script error output:\n", e.stderr.decode())


app = FastAPI(lifespan=lifespan)


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
        return msg.turn in filter_condition.turns
    elif filter_condition.type == "id":
        return msg.agent_id in filter_condition.ids
    elif filter_condition.type == "name":
        return msg.name.lower() in [name.lower() for name in filter_condition.names]
    return False


def filter_messages(
    msgs: List[MessageUnit], filter_condition: Optional[FilterCondition]
) -> List[MessageUnit]:
    return [msg for msg in msgs if check_msg_filter(msg, filter_condition)]


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    filter_condition = None

    async def send_filtered_history(filter_condition: Optional[FilterCondition]):
        async with lock:
            msgs = message_manager.messages.copy()
        msgs = filter_messages(msgs, filter_condition)
        for msg in msgs:
            await manager.send(msg)

    try:
        while True:
            # get msg from message queue
            try:
                msg = message_manager.message_queue.get_nowait()
            except Empty:
                msg = None

            # listen for filter condition without timeout
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
                filter_condition = json.loads(data)
                logger.info(f"Receive new filter condition: {filter_condition}")
                filter_condition = FilterCondition(**filter_condition)
                if filter_condition.type == "None":
                    filter_condition = None
                await send_filtered_history(filter_condition)
            except asyncio.TimeoutError:
                pass

            # send msg to frontend
            if msg and (
                not filter_condition or check_msg_filter(msg, filter_condition)
            ):
                await manager.send(msg)
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    finally:
        await manager.disconnect(websocket)


# @app.websocket("/ws/{id}")
# async def websocket_agent_endpoint(websocket: WebSocket, id: int):
# await manager.connect(websocket, id)
# try:
#     while True:
#         data = await websocket.receive_text()
#         # TODO
# except WebSocketDisconnect:
#     manager.disconnect(websocket)
# finally:
#     manager.disconnect(websocket)


@app.get("/scene", response_model=List[Scene])
def get_scenes():
    assets_path = os.path.join(proj_path, "assets")
    scenes = []
    for scene in os.listdir(assets_path):
        scene_path = os.path.join(assets_path, scene)
        if os.path.isdir(scene_path):
            with open(os.path.join(scene_path, "desc.txt"), "r") as f:
                desc = f.read()
            pic_path = os.path.join(scene_path, "pic.png")
            print(desc)
            print(pic_path)
            scenes.append(Scene(name=scene, desc=desc, pic_path=pic_path))
    return scenes


@app.put("/scene")
def put_scene(scene_name: str):
    global _scene
    _scene = scene_name
    return {"status": "success"}


@app.get("/agents")
def get_agents(query: Optional[str] = None):
    agents = simulator.agents

    def fuzzy_search(agents, query):
        return [agent for agent in agents if query.lower() in agent.name.lower()]

    if query:
        agents = fuzzy_search(agents, query)
    return [try_serialize_dict(agent.__dict__) for agent in agents]


@app.get("/agent/config", response_model=List[AgentConfig])
def get_agent_config():
    configs_path = Path(
        os.path.join(proj_path, "simulation", "examples", _scene, "configs")
    )
    all_agent_configs = configs_path.glob("all_*_agent_configs.json")
    resp = []
    for agent_config in all_agent_configs:
        with open(agent_config, "r") as f:
            agent_config = json.load(f)
            agent_cls = {
                "class": agent_config[0]["class"],
                "num_agents": len(agent_config),
            }
            print(agent_cls)
            resp.append(AgentConfig(**agent_cls))
    return resp


@app.put("/agent/config")
def put_agent_config(req: List[AgentConfig]):
    req = {agent.cls: agent.num_agents for agent in req}
    configs_path = Path(
        os.path.join(proj_path, "simulation", "examples", _scene, "configs")
    )
    all_agent_configs = configs_path.glob("all_*_agent_configs.json")
    for all_agent_config in all_agent_configs:
        with open(all_agent_config, "r") as f:
            agent_configs = json.load(f)
            agent_num = req[agent_configs[0]["class"]]
            agent_configs = random.choices(agent_configs, k=agent_num)
            agent_configs_path = os.path.join(
                configs_path, all_agent_config.name.removeprefix("all_")
            )
            with open(agent_configs_path, "w") as agent_config_file:
                json.dump(
                    agent_configs, agent_config_file, ensure_ascii=False, indent=4
                )
    return {"status": "success"}


@app.get("/agent/{id}")
def get_agent(id: int):
    agents = simulator.agents
    try:
        return try_serialize_dict(agents[id].__dict__)
    except IndexError:
        return HTMLResponse(content="Agent not found.", status_code=404)


@app.put("/agent/{id}")
def put_agent(id: int, new_agent):
    agents = simulator.agents
    try:
        agent = agents[id]
        agent.update_from_dict(new_agent)
        return {"status": "success"}
    except IndexError:
        return HTMLResponse(content="Agent not found.", status_code=404)


@app.get("/model", response_model=List[ModelConfig])
async def get_model_configs():
    config_file = os.path.join(
        proj_path, "simulation", "examples", _scene, "configs", "model_configs.json"
    )
    logger.info(f"Get model config from {config_file}")
    async with aiofiles.open(config_file, "r") as f:
        model_configs = await f.read()
        model_configs = json.loads(model_configs)
        return model_configs


@app.put("/model", response_model=List[ModelConfig])
async def put_model_configs(model_configs: List[ModelConfig]):
    config_file = os.path.join(
        proj_path, "simulation", "examples", _scene, "configs", "model_configs.json"
    )
    logger.info(f"Put model config to {config_file}")
    async with aiofiles.open(config_file, "w") as f:
        await f.write(model_configs)
    return {"status": "success"}


@app.get("/memory", response_model=List[MemoryConfig])
async def get_memory_config():
    all_memory_configs = [
        {
            "class": memory.__name__,
            "args": inspect.getfullargspec(memory.__init__).kwonlydefaults,
        }
        for memory in [
            NoneMemory,
            ShortMemory,
            ShortLongMemory,
            ShortLongReflectionMemory,
        ]
    ]
    return all_memory_configs


@app.put("/memory")
async def put_memory_config(memory_config: MemoryConfig):
    config_file = os.path.join(
        proj_path, "simulation", "examples", _scene, "configs", "memory_configs.json"
    )
    logger.info(f"Put memory config to {config_file}")
    async with aiofiles.open(config_file, "w") as f:
        await f.write(memory_config)
    return {"status": "success"}


# @app.get("/checkpoint/{scene}", response_model=List[CheckpointResp])
# def get_checkpoint(scene: str = "job_seeking"):
#     # Clear restore_file_path in simulation_config.yml
#     simulation_config_path = os.path.join(
#         proj_path, "simulation", "examples", scene, "configs", "simulation_config.yml"
#     )
#     with open(simulation_config_path, "r") as f:
#         simulation_config = yaml.safe_load(f)
#     simulation_config["restore_file_path"] = None
#     with open(simulation_config_path, "w") as f:
#         yaml.safe_dump(simulation_config, f)

#     runs = Path(os.path.join(proj_path, "runs"))
#     if not runs.exists():
#         runs.mkdir()
#     checkpoints = runs.glob("*/*.pkl")
#     run_names = [checkpoint.parent for checkpoint in checkpoints]
#     resp = []
#     for run_name in run_names:
#         checkpoints = run_name.glob("*.pkl")
#         resp.append(
#             CheckpointResp(
#                 run_name=run_name.name,
#                 pkls=[checkpoint.name for checkpoint in checkpoints],
#             )
#         )
#     return resp


@app.post("/checkpoint")
def load_checkpoint(checkpoint_req: PathReq):
    logger.info(f"Load checkpoint from {checkpoint_req.path}")
    checkpoint_path = checkpoint_req.path
    simulation_config_path = os.path.join(
        proj_path, "simulation", "examples", _scene, "configs", "simulation_config.yml"
    )
    with open(simulation_config_path, "r") as f:
        simulation_config = yaml.safe_load(f)
    simulation_config["load_simulator_path"] = checkpoint_path
    with open(simulation_config_path, "w") as f:
        yaml.safe_dump(simulation_config, f)
    return {"status": "success"}


@app.get("/savedir", response_model=PathReq)
def get_savedir():
    simulation_config_path = os.path.join(
        proj_path, "simulation", "examples", _scene, "configs", "simulation_config.yml"
    )
    with open(simulation_config_path, "r") as f:
        simulation_config = yaml.safe_load(f)
    return PathReq(path=simulation_config["save_dir"])


@app.put("/savedir")
def put_savedir(req: PathReq):
    savedir = req.path
    simulation_config_path = os.path.join(
        proj_path, "simulation", "examples", _scene, "configs", "simulation_config.yml"
    )
    with open(simulation_config_path, "r") as f:
        simulation_config = yaml.safe_load(f)
    simulation_config["save_dir"] = savedir
    with open(simulation_config_path, "w") as f:
        yaml.safe_dump(simulation_config, f)
    return {"status": "success"}


@app.post("/distributed")
def configure_distributed(req: DistributedConfig):
    global distributed, distributed_args
    simulation_config_path = os.path.join(
        proj_path, "simulation", "examples", _scene, "configs", "simulation_config.yml"
    )
    with open(simulation_config_path, "r") as f:
        simulation_config = yaml.safe_load(f)
    simulation_config["distributed"] = distributed = req.distributed
    with open(simulation_config_path, "w") as f:
        yaml.safe_dump(simulation_config, f)
    distributed_args = req.args
    return {"status": "success"}


@app.get("/messages", response_model=List[MessageUnit])
async def get_messages(filter_condition: Optional[FilterCondition] = None):
    async with lock:
        msgs = message_manager.messages.copy()
    return filter_messages(msgs, filter_condition)


@app.post("/start")
async def start():
    # Distributed setup
    if distributed:
        # assign host and port for agents
        module_path = f"simulation.examples.{_scene}.assign_host_port"
        assign_host_port = importlib.import_module(module_path).main
        assign_host_port(distributed_args)

        # launch server
        launch_server_sh_path = os.path.join(
            proj_path, "simulation", "examples", _scene, "launch_server.sh"
        )
        command = [
            "bash",
            launch_server_sh_path,
            distributed_args.server_num_per_host,
            distributed_args.base_port,
        ]
        try:
            result = subprocess.run(
                command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            logger.info("Launch server script output:\n", result.stdout.decode())
        except subprocess.CalledProcessError as e:
            logger.error("Launch server script failed with return code:", e.returncode)
            logger.error("Launch server script error output:\n", e.stderr.decode())

    module_path = f"simulation.examples.{_scene}.simulator"
    Simulator = importlib.import_module(module_path).Simulator
    global simulator
    simulator = Simulator()
    simulation_thread = Thread(target=simulator.run)
    simulation_thread.start()
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
