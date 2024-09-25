import asyncio
from concurrent import futures
import json
import os
import random
import importlib
import inspect
from pathlib import Path
from queue import Empty, Queue
import re
from threading import Thread, Event
import threading
import time
from typing import Dict, List, Literal, Optional, Union
from concurrent.futures import ThreadPoolExecutor
from starlette.websockets import WebSocketState

import aiofiles
from ruamel.yaml import YAML
from agentscope.message import Msg
from contextlib import asynccontextmanager
from fastapi import Body, FastAPI, WebSocket, WebSocketDisconnect, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from backend.utils.connection import ChatMessage, manager
from simulation.helpers.message import message_manager, MessageUnit, StateUnit
from simulation.helpers.events import (
    play_event,
    stop_event,
    kill_event,
    pause_success_event,
)
from backend.utils.body_models import (
    Scene,
    ModelConfig,
    AgentConfig,
    MemoryConfig,
    PathReq,
    FilterCondition,
    DistributedArgs,
    DistributedConfig,
    BroadcastMsg,
    Coord,
    AgentInfo,
    AgentStateInfo,
    GPTReq,
    ChangedMsg,
    Transform,
)
from simulation.memory import (
    NoneMemory,
    ShortMemory,
    ShortLongMemory,
    ShortLongReflectionMemory,
)
from backend.utils.utils import run_sh_async, run_sh_blocking, run_sh_train_blocking
from backend.utils.sample import generate_points_sampling
from backend.chatgpt_api import rewritten_responses, rate_responses


yaml = YAML()
executor = ThreadPoolExecutor()
proj_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
templates = Jinja2Templates(directory=os.path.join(proj_path, "backend", "templates"))
assets_files = StaticFiles(directory=os.path.join(proj_path, "assets"))


_scene = "job_seeking"
events: Dict[str, Event] = {}
queue = Queue()
simulator = None
simulation_thread: Thread = None
lock = threading.RLock()
distributed: bool = True
cur_msgs: List[MessageUnit] = None
backend_server_url: str = None
agent_coordinates: Dict[str, List[float]] = {}
favorite_agents = []
transform: Transform = Transform()
avatar_radius = 0.0001
agent_info = {}
train_progress = -1


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Update backend_server_url
    global backend_server_url
    host = os.environ.get("HOST", "0.0.0.0")
    port = os.environ.get("PORT", 9000)
    backend_server_url = f"http://{host}:{port}"
    # Launch LLM
    launch_llm_sh_path = os.path.join(proj_path, "exp2", "scripts", "launch_llm.sh")
    run_sh_async(launch_llm_sh_path, "8084", "2")

    yield

    # Kill LLM
    kill_llm_sh_path = os.path.join(proj_path, "exp2", "scripts", "kill_llm.sh")
    run_sh_blocking(kill_llm_sh_path)

    # Clean distributed servers
    if distributed:
        kill_server_sh_path = os.path.join(
            proj_path, "simulation", "examples", _scene, "kill_all_server.sh"
        )
        run_sh_blocking(kill_server_sh_path)


app = FastAPI(lifespan=lifespan)
app.mount("/assets", assets_files, name="assets")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def check_filter(
    msg_or_state: Union[MessageUnit, StateUnit],
    filter_condition: Optional[FilterCondition],
) -> bool:
    if filter_condition is None or filter_condition.condition == "None":
        return True
    elif filter_condition.condition == "id":
        return msg_or_state.agent_id in filter_condition.ids
    elif filter_condition.condition == "name":
        return msg_or_state.name.lower() in [
            name.lower() for name in filter_condition.names
        ]
    elif filter_condition.condition == "type":
        return msg_or_state.agent_type.lower() in [
            type.lower() for type in filter_condition.types
        ]
    return False


def filter_msgs_or_states(
    msgs_or_states: List[Union[MessageUnit, StateUnit]],
    filter_condition: Optional[FilterCondition],
) -> List[Union[MessageUnit, StateUnit]]:
    return [
        msg_or_state
        for msg_or_state in msgs_or_states
        if check_filter(msg_or_state, filter_condition)
    ]


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    filter_condition = None

    try:
        while True:
            # get state from state queue
            try:
                state = message_manager.state_queue.get_nowait()
            except Empty:
                state = None

            # listen for filter condition without timeout
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=1)
                filter_condition = json.loads(data)
                logger.info(f"Receive new filter condition: {filter_condition}")
                filter_condition = FilterCondition(**filter_condition)
                if filter_condition.condition == "None":
                    filter_condition = None
            except asyncio.TimeoutError:
                pass

            # send state to frontend
            if state and (
                not filter_condition or check_filter(state, filter_condition)
            ):
                await manager.send(state)
    except WebSocketDisconnect:
        logger.info("WebSocket /ws disconnected")
    finally:
        if websocket.client_state == WebSocketState.CONNECTED:
            await manager.disconnect(websocket)


@app.websocket("/train")
async def websocket_train_endpoint(websocket: WebSocket):
    global train_progress
    await websocket.accept()
    cur_progress = train_progress
    train_thread = None
    try:
        await websocket.send_json({"train_progress": train_progress})
        while True:
            await asyncio.sleep(0.001)

            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.001)
                logger.info(f"Receive train message: {data}")
                if data == "exit":
                    cancel_tune_sh_path = os.path.join(proj_path, "exp2", "scripts", "cancel_tune.sh")
                    run_sh_blocking(cancel_tune_sh_path)
                    train_thread = None
                    # Launch LLM
                    launch_llm_sh_path = os.path.join(
                        proj_path, "exp2", "scripts", "launch_llm.sh"
                    )
                    run_sh_async(launch_llm_sh_path, "8084", "2")
                    train_progress = cur_progress = -1
                    await websocket.send_json({"train_progress": train_progress})
                elif data == "rewrite" or data == "rate":
                    # Kill LLM
                    kill_llm_sh_path = os.path.join(proj_path, "exp2", "scripts", "kill_llm.sh")
                    run_sh_blocking(kill_llm_sh_path)
                    train_progress = cur_progress = 0
                    await websocket.send_json({"train_progress": train_progress})
                    tune_llm_sh_path = os.path.join(proj_path, "exp2", "scripts", "tune_llm.sh")
                    if data == "rewrite":
                        train_thread = Thread(target=run_sh_train_blocking, args=(tune_llm_sh_path, "sft"))
                    elif data == "rate":
                        train_thread = Thread(target=run_sh_train_blocking, args=(tune_llm_sh_path, "ppo"))
                    train_thread.start()
            except asyncio.TimeoutError:
                pass

            if train_progress != cur_progress:
                cur_progress = train_progress
                await websocket.send_json({"train_progress": train_progress})
            if train_thread is not None and not train_thread.is_alive():
                # Launch LLM
                launch_llm_sh_path = os.path.join(
                    proj_path, "exp2", "scripts", "launch_llm.sh"
                )
                run_sh_async(launch_llm_sh_path, "8084", "2")
                train_thread = None

                # Reset agents' model.model_name
                if simulator is not None:
                    agents = simulator.agents
                    results = []
                    for agent in agents:
                        results.append(agent.set_attr("model.model_name", "lora"))
                    for res in results:
                        res.result()
                train_progress = cur_progress = -1
                await websocket.send_json({"train_progress": train_progress})
    except WebSocketDisconnect:
        logger.info("WebSocket /train disconnected")
        if train_thread is not None and not train_thread.is_alive():
            # Launch LLM
            launch_llm_sh_path = os.path.join(
                proj_path, "exp2", "scripts", "launch_llm.sh"
            )
            run_sh_async(launch_llm_sh_path, "8084", "2")
            train_thread = None

            # Reset agents' model.model_name
            if simulator is not None:
                agents = simulator.agents
                results = []
                for agent in agents:
                    results.append(agent.set_attr("model.model_name", "lora"))
                for res in results:
                    res.result()
            train_progress = cur_progress = -1
            await websocket.send_json({"train_progress": train_progress})
    finally:
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close()


@app.websocket("/round")
async def websocket_round_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        cur_round = -1
        if simulator is not None:
            cur_round = simulator.cur_round
            await websocket.send_json({"round": cur_round})
        while True:
            await asyncio.sleep(1)
            if simulator is not None and cur_round != simulator.cur_round:
                cur_round = simulator.cur_round
                await websocket.send_json({"round": cur_round})
    except WebSocketDisconnect:
        logger.info("WebSocket /round disconnected")
    finally:
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close()


@app.websocket("/chat/{id}")
async def websocket_chat_endpoint(websocket: WebSocket, id: str):
    await manager.connect(websocket, id)
    try:
        send_msgs = [
            msg.model_dump_json() for msg in manager.agent_connections_history[id]
        ]
        await manager.agent_connections[id].send_text(f"[{', '.join(send_msgs)}]")
        env = None
        if simulator is not None:
            env = simulator.env
        while True:
            data = await websocket.receive_text()
            logger.info(f"Receive chat message: {data}")
            manager.agent_connections_history[id].append(
                ChatMessage(sender="human", message=data)
            )
            if data == "exit":
                break
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(executor, env.interview, id, data)
            # resp = env.interview(id, data)
            logger.info(f"Send chat message: {resp}")
            await manager.send_to_agent(id, resp)
    except WebSocketDisconnect:
        logger.info(f"WebSocket /chat/{id} disconnected")
    finally:
        if websocket.client_state == WebSocketState.CONNECTED:
            await manager.disconnect(websocket, id)


@app.get("/current-scene")
def get_current_scene():
    return _scene


@app.get("/scene", response_model=List[Scene])
def get_scenes():
    assets_path = os.path.join(proj_path, "assets", "scenes")
    scenes = []
    for scene in os.listdir(assets_path):
        scene_path = os.path.join(assets_path, scene)
        if os.path.isdir(scene_path):
            with open(os.path.join(scene_path, "desc.txt"), "r") as f:
                desc = f.read()
            pic_path = os.path.join("/assets", scene, "pic.png")
            scenes.append(Scene(name=scene, desc=desc, pic_path=pic_path))
    return scenes


@app.put("/scene")
def put_scene(scene_name: str):
    global _scene
    _scene = scene_name
    return HTMLResponse()


@app.get("/transform", response_model=Transform)
def get_transform():
    return transform


@app.put("/transform")
def put_transform(req: Transform):
    global transform
    transform = req
    return HTMLResponse()


@app.get("/agents", response_model=List[AgentInfo])
def get_agents(
    type: Optional[str] = None,
    query: Optional[str] = None,
    favorite: bool = False,
):
    if simulator is None:
        return []
    if favorite:
        agents = favorite_agents
    else:
        agents = simulator.agents

    if query:
        agents = [
            agent
            for agent in agents
            if (not type or agent_info[agent.agent_id]["cls"] == type)
            and query.lower() in agent_info[agent.agent_id]["name"].lower()
        ]
    elif type:
        agents = [
            agent for agent in agents if agent_info[agent.agent_id]["cls"] == type
        ]

    resp = []
    # tasks = []
    # with futures.ThreadPoolExecutor() as executor:
    #     for agent in agents:
    #         tasks.append(executor.submit(agent.get_attr, "state"))
    #     for idx, task in enumerate(tasks):
    #         state = task.result()
    #         agent_info[agents[idx].agent_id]["state"] = state
    for agent in agents:
        resp.append(
            AgentInfo(
                name=agent_info[agent.agent_id]["name"],
                id=agent.agent_id,
                cls=agent_info[agent.agent_id]["cls"],
                state="idle",
                profile=agent_info[agent.agent_id]["profile"],
                gender=agent_info[agent.agent_id]["gender"],
                coordinates=Coord(
                    x=agent_coordinates[agent.agent_id][0],
                    y=agent_coordinates[agent.agent_id][1],
                ),
                avatar=agent_info[agent.agent_id]["avatar_path"],
            )
        )
    return resp


# @app.get("/agents")
# def get_agents(query: Optional[str] = None):
#     agents = simulator.agents

#     def fuzzy_search(agents, query):
#         return [
#             agent
#             for agent in agents
#             if query.lower() in agent.name.lower() or query == str(agent.id)
#         ]

#     if query:
#         agents = fuzzy_search(agents, query)
#     return [try_serialize_dict(agent.__dict__) for agent in agents]


@app.get("/agent/config", response_model=List[AgentConfig])
def get_agent_config():
    configs_path = Path(
        os.path.join(proj_path, "simulation", "examples", _scene, "configs")
    )
    all_agent_configs = configs_path.glob("all_*_configs.json")
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
    all_agent_configs = configs_path.glob("all_*_configs.json")
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
    return HTMLResponse()


# @app.get("/agent/config", response_model=List[str])
# def get_agent_classes_config():
#     agent_module = importlib.import_module(f"simulation.examples.{_scene}.agent")
#     agent_classes = inspect.getmembers(agent_module, inspect.isclass)
#     resp = [agent_cls[0] for agent_cls in agent_classes]
#     # configs_path = Path(
#     #     os.path.join(proj_path, "simulation", "examples", _scene, "configs")
#     # )
#     # all_agent_configs = configs_path.glob("all_*_agent_configs.json")
#     # resp = []
#     # for agent_config in all_agent_configs:
#     #     with open(agent_config, "r") as f:
#     #         agent_config = json.load(f)
#     #         agent_cls = {
#     #             "class": agent_config[0]["class"],
#     #             "num_agents": len(agent_config),
#     #         }
#     #         print(agent_cls)
#     #         resp.append(AgentConfig(**agent_cls))
#     return resp


# @app.put("/agent/config")
# def put_agent_config(req: AgentConfig):
#     configs_path = os.path.join(proj_path, "simulation", "examples", _scene, "configs")
#     profile_path = os.path.join(configs_path, f"all_{req.cls}_configs.json")
#     with open(profile_path, "r") as f:
#         agent_configs = json.load(f)
#         agent_configs = random.choices(agent_configs, k=req.num_agents)
#         agent_configs_path = os.path.join(configs_path, f"{req.cls}_configs.json")
#         with open(agent_configs_path, "w") as agent_config_file:
#             json.dump(agent_configs, agent_config_file, ensure_ascii=False, indent=4)
#     return HTMLResponse()


# @app.post("/agent/profile/{cls}", response_model=AgentConfig)
# async def post_agent_profile(cls: str, profile: UploadFile):
#     profile_path = os.path.join(
#         proj_path,
#         "simulation",
#         "examples",
#         _scene,
#         "configs",
#         f"all_{cls}_configs.json",
#     )
#     async with aiofiles.open(profile_path, "wb") as f:
#         await f.write(await profile.read())
#     with open(profile_path, "r") as f:
#         agent_configs = json.load(f)
#         num_agents = len(agent_configs)
#     return AgentConfig(**{"class": cls, "num_agents": num_agents})


@app.get("/agent/favorite", response_model=List[AgentInfo])
def get_favorite_agents():
    if simulator is None:
        return []

    resp = []
    for agent in favorite_agents:
        resp.append(
            AgentInfo(
                name=agent_info[agent.agent_id]["name"],
                id=agent.agent_id,
                cls=agent_info[agent.agent_id]["cls"],
                state=agent.get_attr("state"),
                profile=agent_info[agent.agent_id]["profile"],
                gender=agent_info[agent.agent_id]["gender"],
                coordinates=Coord(
                    x=agent_coordinates[agent.agent_id][0],
                    y=agent_coordinates[agent.agent_id][1],
                ),
                avatar=agent_info[agent.agent_id]["avatar_path"],
            )
        )
    return resp


@app.get("/agent/favorite/{id}")
def get_favorite_agent(id: str):
    if simulator is None:
        return HTMLResponse(content="Simulator is not running.", status_code=400)
    if id in [agent.agent_id for agent in favorite_agents]:
        return True
    else:
        return False


@app.post("/agent/favorite/{id}")
def post_favorite_agent(id: str):
    global favorite_agents
    if simulator is None:
        return HTMLResponse(content="Simulator is not running.", status_code=400)
    agents = simulator.agents
    for agent in agents:
        if agent.agent_id == id:
            favorite_agents.append(agent)
            return HTMLResponse()
    return HTMLResponse(content="Agent not found.", status_code=404)


@app.delete("/agent/favorite/{id}")
def delete_favorite_agent(id: str):
    if simulator is None:
        return HTMLResponse(content="Simulator is not running.", status_code=400)
    global favorite_agents
    favorite_agents = [agent for agent in favorite_agents if agent.agent_id != id]
    return HTMLResponse()


@app.get("/agent/{id}", response_model=AgentInfo)
def get_agent(id: str):
    if simulator is not None:
        agents = simulator.agents
        for agent in agents:
            if agent.agent_id == id:
                return AgentInfo(
                    name=agent_info[id]["name"],
                    id=id,
                    cls=agent_info[agent.agent_id]["cls"],
                    state=agent.get_attr("state"),
                    profile=agent_info[id]["profile"],
                    gender=agent_info[id]["gender"],
                    coordinates=Coord(
                        x=agent_coordinates[agent.agent_id][0],
                        y=agent_coordinates[agent.agent_id][1],
                    ),
                    avatar=agent_info[id]["avatar_path"],
                )
    return HTMLResponse(content="Agent not found.", status_code=404)


# @app.put("/agent/{id}")
# def put_agent(id: int, new_agent):
#     agents = simulator.agents
#     try:
#         agent = agents[id]
#         agent.update_from_dict(new_agent)
#         return HTMLResponse()
#     except IndexError:
#         return HTMLResponse(content="Agent not found.", status_code=404)


@app.post("/broadcast")
def post_broadcast(broadcast_msg: BroadcastMsg):
    if simulator is None:
        return HTMLResponse(content="Simulator is not running.", status_code=400)
    env = simulator.env
    env.broadcast(broadcast_msg.msg)
    return HTMLResponse()


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


@app.put("/model")
def put_model_configs(model_configs: List[ModelConfig]):
    config_file = os.path.join(
        proj_path, "simulation", "examples", _scene, "configs", "model_configs.json"
    )
    logger.info(f"Put model config to {config_file}")
    with open(config_file, "w") as f:
        json.dump(
            [config.model_dump() for config in model_configs],
            f,
            ensure_ascii=False,
            indent=4,
        )
    return HTMLResponse()


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
def put_memory_config(memory_config: MemoryConfig):
    config_file = os.path.join(
        proj_path, "simulation", "examples", _scene, "configs", "memory_configs.json"
    )
    logger.info(f"Put memory config to {config_file}")
    with open(config_file, "w") as f:
        json.dump(memory_config.model_dump(), f, ensure_ascii=False, indent=4)
    return HTMLResponse()


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


@app.get("/checkpoint")
def get_current_checkpoint():
    simulation_config_path = os.path.join(
        proj_path, "simulation", "examples", _scene, "configs", "simulation_config.yml"
    )
    with open(simulation_config_path, "r") as f:
        simulation_config = yaml.load(f)
    return PathReq(path=simulation_config["load_simulator_path"])


@app.put("/checkpoint")
def load_checkpoint(checkpoint_req: PathReq):
    logger.info(f"Load checkpoint from {checkpoint_req.path}")
    checkpoint_path = checkpoint_req.path
    simulation_config_path = os.path.join(
        proj_path, "simulation", "examples", _scene, "configs", "simulation_config.yml"
    )
    with open(simulation_config_path, "r") as f:
        simulation_config = yaml.load(f)
    simulation_config["load_simulator_path"] = checkpoint_path
    with open(simulation_config_path, "w") as f:
        yaml.dump(simulation_config, f)
    return HTMLResponse()


@app.get("/savedir", response_model=PathReq)
def get_savedir():
    simulation_config_path = os.path.join(
        proj_path, "simulation", "examples", _scene, "configs", "simulation_config.yml"
    )
    with open(simulation_config_path, "r") as f:
        simulation_config = yaml.load(f)
    return PathReq(path=simulation_config["save_dir"])


@app.put("/savedir")
def put_savedir(req: PathReq):
    savedir = req.path
    simulation_config_path = os.path.join(
        proj_path, "simulation", "examples", _scene, "configs", "simulation_config.yml"
    )
    with open(simulation_config_path, "r") as f:
        simulation_config = yaml.load(f)
    simulation_config["save_dir"] = savedir
    with open(simulation_config_path, "w") as f:
        yaml.dump(simulation_config, f)
    return HTMLResponse()


@app.get("/distributed", response_model=DistributedConfig)
def get_distributed_config():
    simulation_config_path = os.path.join(
        proj_path, "simulation", "examples", _scene, "configs", "simulation_config.yml"
    )
    with open(simulation_config_path, "r") as f:
        simulation_config = yaml.load(f)
    return DistributedConfig(
        distributed=True,
        args=DistributedArgs(
            host=simulation_config["host"],
            base_port=simulation_config["base_port"],
            server_num_per_host=simulation_config["server_num_per_host"],
        ),
    )


@app.put("/distributed")
def put_distributed_config(req: DistributedConfig):
    global distributed
    distributed = req.distributed
    simulation_config_path = os.path.join(
        proj_path, "simulation", "examples", _scene, "configs", "simulation_config.yml"
    )
    with open(simulation_config_path, "r") as f:
        simulation_config = yaml.load(f)
    simulation_config["base_port"] = req.args.base_port
    simulation_config["server_num_per_host"] = req.args.server_num_per_host
    with open(simulation_config_path, "w") as f:
        yaml.dump(simulation_config, f)
    return HTMLResponse()


@app.get("/states", response_model=List[AgentStateInfo])
def get_all_agent_states_info():
    module_path = f"simulation.examples.{_scene}.agent"
    all_agent_states_info = importlib.import_module(module_path).ALL_AGENT_STATES
    return [
        AgentStateInfo(agent_cls_name=agent_cls_name, states=states)
        for agent_cls_name, states in all_agent_states_info.items()
    ]


@app.get("/messages", response_model=List[MessageUnit])
def get_messages_with_filter(
    # filter_condition: Optional[FilterCondition] = None,
    offset: Optional[int] = 0,
    limit: Optional[int] = 10,
):
    global cur_msgs
    if cur_msgs is None:
        with lock:
            cur_msgs = message_manager.messages.copy()
            cur_msgs = [msg for msg in cur_msgs if msg.msg_id]
    filter_condition = None
    msgs = filter_msgs_or_states(cur_msgs, filter_condition)
    return msgs[offset : offset + limit]


def change_msgs(new_msgs: List[ChangedMsg], mode: Optional[Literal["rewrite", "rate"]] = None):
    with lock:
        for new_msg in new_msgs:
            if mode == "rewrite":
                message_manager.messages[new_msg.msg_id].rewritten_response = (
                    new_msg.rewritten_response
                )
            elif mode == "rate":
                message_manager.messages[new_msg.msg_id].rating = new_msg.rating
            else:
                message_manager.messages[new_msg.msg_id].rewritten_response = (
                    new_msg.rewritten_response
                )
                message_manager.messages[new_msg.msg_id].rating = new_msg.rating
    global cur_msgs
    cur_msgs_ids = [msg.msg_id for msg in cur_msgs]
    for new_msg in new_msgs:
        if new_msg.msg_id in cur_msgs_ids:
            if mode == "rewrite":
                cur_msgs[cur_msgs_ids.index(new_msg.msg_id)].rewritten_response = (
                    new_msg.rewritten_response
                )
            elif mode == "rate":
                cur_msgs[cur_msgs_ids.index(new_msg.msg_id)].rating = new_msg.rating
            else:
                cur_msgs[cur_msgs_ids.index(new_msg.msg_id)].rewritten_response = (
                    new_msg.rewritten_response
                )
                cur_msgs[cur_msgs_ids.index(new_msg.msg_id)].rating = new_msg.rating


@app.put("/messages/random")
def random_selection_messages(num: int):
    global cur_msgs
    with lock:
        cur_msgs = message_manager.messages.copy()
        cur_msgs = [msg for msg in cur_msgs if msg.msg_id]
    if num != 0:
        cur_msgs = sorted(random.choices(cur_msgs, k=num), key=lambda x: x.msg_id)
    return HTMLResponse()


@app.post("/messages/random/undo")
def undo_random_selection():
    global cur_msgs
    with lock:
        cur_msgs = message_manager.messages.copy()
        cur_msgs = [msg for msg in cur_msgs if msg.msg_id]
    return HTMLResponse()


@app.put("/messages")
def save_changed_messages(msgs: List[ChangedMsg]):
    change_msgs(msgs)
    return HTMLResponse()


@app.post("/gpt")
def chatgpt(req: GPTReq):
    with lock:
        msgs = message_manager.messages.copy()
    msgs = [msgs[id].model_dump(include={"prompt", "completion", "selection_num"}) for id in req.msg_ids]
    if req.mode == "rewrite":
        resps = rewritten_responses(msgs)
        change_msgs(
            [
                ChangedMsg(msg_id=msg_id, rewritten_response=resps[idx])
                for idx, msg_id in enumerate(req.msg_ids)
            ],
            "rewrite",
        )
    elif req.mode == "rate":
        resps = rate_responses(msgs)
        change_msgs(
            [
                ChangedMsg(msg_id=msg_id, rating=resps[idx])
                for idx, msg_id in enumerate(req.msg_ids)
            ],
            "rate",
        )
    return HTMLResponse()


@app.post("/tune/{mode}")
def tune(mode: Literal["rewrite", "rate"]):

    # Kill LLM
    kill_llm_sh_path = os.path.join(proj_path, "exp2", "scripts", "kill_llm.sh")
    run_sh_blocking(kill_llm_sh_path)

    # Tune LLM
    tune_llm_sh_path = os.path.join(proj_path, "exp2", "scripts", "tune_llm.sh")
    if mode == "rewrite":
        tuning_mode = "sft"
    elif mode == "rate":
        tuning_mode = "ppo"
    run_sh_train_blocking(tune_llm_sh_path, tuning_mode)

    # Launch LLM
    launch_llm_sh_path = os.path.join(
        proj_path, "exp2", "scripts", "launch_llm.sh"
    )
    run_sh_async(launch_llm_sh_path, "8084", "2")

    # Reset agents' model.model_name
    agents = simulator.agents
    results = []
    for agent in agents:
        results.append(agent.set_attr("model.model_name", "lora"))
    for res in results:
        res.result()

    return HTMLResponse()


@app.post("/export")
def export_changed_messages(mode: Literal["rewrite", "rate"]):
    with lock:
        msgs = message_manager.messages.copy()
    if mode == "rewrite":
        msgs = [
            {"prompt": msg.prompt, "completion": msg.rewritten_response}
            for msg in msgs
            if msg.rewritten_response
        ]
        export_path = os.path.join(
            proj_path, "exp2", "datasets", "sft_data", "sft_data.json"
        )
        with open(export_path, "w") as f:
            json.dump(msgs, f, ensure_ascii=False, indent=4)
    elif mode == "rate":
        msgs = [
            {"prompt": msg.prompt, "completion": msg.completion, "reward": msg.rating}
            for msg in msgs
            if msg.rating
        ]
        export_path = os.path.join(
            proj_path, "exp2", "datasets", "ppo_data", "ppo_data.json"
        )
        with open(export_path, "w") as f:
            json.dump(msgs, f, ensure_ascii=False, indent=4)
    return HTMLResponse()


@app.post("/api/state")
def post_state(state: StateUnit):
    message_manager.add_state(state)
    # logger.info(f"state: {state.agent_id} -- {state.state}")
    return HTMLResponse()


@app.post("/api/message")
def post_messages(message: MessageUnit):
    message_manager.add_message(message)
    # logger.info(f"message: {message.agent_id} -- {message.completion}")
    return HTMLResponse()


@app.get("/avatar-radius")
def get_avatar_radius():
    if simulator is None:
        return HTMLResponse(
            content="Simulator is not running. You should start first.",
            status_code=400,
        )
    return {"radius": avatar_radius}


@app.post("/start")
async def start():
    global simulator, simulation_thread, avatar_radius

    # 用于控制循环重试的标志和重试次数
    success = False
    max_retries = 5  # 最大重试次数
    retry_count = 0  # 当前重试次数

    # 开始循环，最多重试 max_retries 次
    while not success and retry_count < max_retries:
        try:
            if simulator is not None:
                return HTMLResponse(
                    content="Simulator is already running. You should reset first.",
                    status_code=400,
                )
            # launch server
            simulation_config_path = os.path.join(
                proj_path, "simulation", "examples", _scene, "configs", "simulation_config.yml"
            )
            with open(simulation_config_path, "r") as f:
                simulation_config = yaml.load(f)
            launch_server_sh_path = os.path.join(
                proj_path, "simulation", "examples", _scene, "launch_server.sh"
            )
            run_sh_blocking(
                launch_server_sh_path,
                str(simulation_config["server_num_per_host"]),
                str(simulation_config["base_port"]),
            )
            time.sleep(10 + 5 * retry_count)

            module_path = f"simulation.examples.{_scene}.simulator"
            Simulator = importlib.import_module(module_path).Simulator
            simulator = Simulator()
            agents = simulator.agents
            results = []
            for agent in agents:
                results.append(agent.set_attr("backend_server_url", backend_server_url))
            for res in results:
                res.result()

            # Parameters
            n_samples = len(agents)  # 需要生成的点数
            canvas_size = 1.0  # 画布的尺寸，此时默认为1*1的
            initial_center_dist = 0.08  # 初始默认圆心距
            radius_ratio = 0.4  # 初始默认半径占圆心距的比例

            # Generate points using Poisson disk sampling
            # points, final_radius = poisson_disk_sampling(width, height, n_samples, initial_radius)
            points, final_radius = generate_points_sampling(
                k=n_samples,
                radius_ratio=radius_ratio,
                initial_center_dist=initial_center_dist,
                canvas_size=canvas_size,
            )
            avatar_radius = final_radius * 0.75
            for idx, agent in enumerate(agents):
                agent_coordinates[agent.agent_id] = list(points[idx])
            manager.all_agents_state = {agent.agent_id: "idle" for agent in agents}

            # AgentInfo(
            #     name=agent.name,
            #     id=agent.agent_id,
            #     cls=agent._init_settings["class_name"],
            #     state=agent.get_attr(attr="state"),
            #     profile=agent.get_attr(attr="_profile"),
            #     gender=gender,
            #     coordinates=Coord(x=agent_coordinates[agent.agent_id][0], y=agent_coordinates[agent.agent_id][1]),
            #     avatar=avatar_path,
            # )
            for agent in agents:
                agent_info[agent.agent_id] = {
                    "cls": agent._init_settings["class_name"],
                }
            tasks = []
            with futures.ThreadPoolExecutor() as executor:
                for agent in agents:
                    tasks.append(executor.submit(agent.get_attr, "name"))
                for idx, task in enumerate(tasks):
                    agent_info[agents[idx].agent_id]["name"] = task.result()
            tasks = []
            with futures.ThreadPoolExecutor() as executor:
                for agent in agents:
                    tasks.append(executor.submit(agent.get_attr, "_profile"))
                for idx, task in enumerate(tasks):
                    agent_info[agents[idx].agent_id]["profile"] = task.result()
            tasks = []
            with futures.ThreadPoolExecutor() as executor:
                for agent in agents:
                    tasks.append(executor.submit(agent.get_attr, "gender"))
                for idx, task in enumerate(tasks):
                    gender = task.result()
                    avatar_path = os.path.join("/assets", "avatar")
                    match = re.search(r"\d", agents[idx].agent_id)
                    num = match.group() if match else 0
                    if gender is None or gender.lower() not in ["female", "male"]:
                        gender = None
                        avatar_path = os.path.join(avatar_path, "none", f"{num}.png")
                    else:
                        gender = gender.lower()
                        avatar_path = os.path.join(avatar_path, gender, f"{num}.png")
                    agent_info[agents[idx].agent_id]["gender"] = gender
                    agent_info[agents[idx].agent_id]["avatar_path"] = avatar_path

            simulation_thread = Thread(target=simulator.run)
            simulation_thread.start()
            success = True
        except Exception as e:
            logger.error(e)
            kill_server_sh_path = os.path.join(
                proj_path, "simulation", "examples", _scene, "kill_all_server.sh"
            )
            run_sh_blocking(kill_server_sh_path)

            # 增加重试计数器
            retry_count += 1
            logger.info(f"Retry attempt {retry_count} of {max_retries}")

            # 如果达到最大重试次数，返回错误
            if retry_count >= max_retries:
                return HTMLResponse(content="Failed to start simulator after multiple attempts.", status_code=500)

    return HTMLResponse(content="Simulator started successfully.", status_code=200)


@app.post("/pause")
async def pause():
    if simulator is None:
        return HTMLResponse(content="Simulator is not running. Please start simulation first.", status_code=400)
    if simulator.cur_round == -1:
        # Distribute MsgID for messages
        with lock:
            for idx, msg in enumerate(message_manager.messages):
                msg.msg_id = idx
        return HTMLResponse()
    if not play_event.is_set():
        return HTMLResponse(content="Simulation is already paused.", status_code=409)
    message_manager.message_queue.put("Pause simulation.")
    pause_success_event.clear()
    play_event.clear()
    logger.info("Pause simulation.")
    if pause_success_event.is_set():
        logger.info("Event is set.")
    else:
        logger.info("Event is not set.")
    while not pause_success_event.is_set():
        await asyncio.sleep(0.1)
    # Distribute MsgID for messages
    with lock:
        for idx, msg in enumerate(message_manager.messages):
            msg.msg_id = idx
    return HTMLResponse()


@app.post("/resume")
async def resume():
    if simulator is None:
        return HTMLResponse(content="Simulator is not running. Please start simulation first.", status_code=400)
    if simulator.cur_round == -1:
        return HTMLResponse(content="Simulation has already finished.", status_code=409)
    if play_event.is_set():
        return HTMLResponse(content="Simulation is already running.", status_code=409)
    global cur_msgs
    cur_msgs = None
    message_manager.clear()
    message_manager.message_queue.put("Resume simulation.")
    play_event.set()
    pause_success_event.clear()
    return HTMLResponse()


@app.post("/stop")
async def stop():
    stop_event.set()
    return HTMLResponse()


@app.post("/reset")
async def reset():
    # Clean distributed servers
    if distributed:
        kill_server_sh_path = os.path.join(
            proj_path, "simulation", "examples", _scene, "kill_all_server.sh"
        )
        run_sh_blocking(kill_server_sh_path)
    global simulator, simulation_thread, cur_msgs, agent_coordinates, favorite_agents, transform, avatar_radius, agent_info, train_progress
    manager.clear()
    agent_info = {}
    transform = Transform()
    avatar_radius = 0.0001
    train_progress = -1
    simulator = None
    kill_event.set()
    play_event.set()
    if simulation_thread is not None:
        simulation_thread.join()
        logger.info("Simulation thread is joined.")
    simulation_thread = None
    cur_msgs = None
    agent_coordinates = {}
    favorite_agents = []
    message_manager.clear()
    play_event.clear()
    stop_event.clear()
    kill_event.clear()
    pause_success_event.clear()
    message_manager.message_queue.put("Reset simulation.")
    return HTMLResponse()


@app.get("/", response_class=HTMLResponse)
async def read_root():
    return templates.TemplateResponse("index.html", {"request": {}})


@app.post("/store_message")
def store_message(save_data_path: str = Body(..., embed=True)):
    data = [x.model_dump() for x in message_manager.messages]
    with open(save_data_path, "w") as f:
        json.dump(data, f, indent=4)
    message_manager.clear()
    logger.info(f"Store message to {save_data_path}")
    return {"status": "success"}
