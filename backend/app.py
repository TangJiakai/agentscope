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
from typing import Dict, List, Optional, Union

import aiofiles
import uvicorn
from ruamel.yaml import YAML
from agentscope.message import Msg
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from backend.utils.connection import manager
from simulation.helpers.message import message_manager, MessageUnit, StateUnit
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
    InterventionMsg,
    AgentInfo,
    AgentStateInfo,
)
from simulation.memory import (
    NoneMemory,
    ShortMemory,
    ShortLongMemory,
    ShortLongReflectionMemory,
)
from backend.utils.utils import try_serialize_dict


yaml = YAML()
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
cur_msgs: List[MessageUnit] = None
cur_msgs_index: List[int] = None


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
    if filter_condition is None:
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
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
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
        await manager.disconnect(websocket)
    finally:
        await manager.disconnect(websocket)


@app.websocket("/chat/{id}")
async def websocket_chat_endpoint(websocket: WebSocket, id: int):
    await manager.connect(websocket, id)
    try:
        agent = simulator.agents[id]
        while True:
            data = await websocket.receive_text()
            logger.info(f"Receive chat message: {data}")
            await manager.send_to_agent(id, agent.interview(data))
            if data == "exit":
                break
    except WebSocketDisconnect:
        await manager.disconnect(websocket, id)
    finally:
        await manager.disconnect(websocket, id)


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
            scenes.append(Scene(name=scene, desc=desc, pic_path=pic_path))
    return scenes


@app.put("/scene")
def put_scene(scene_name: str):
    global _scene
    _scene = scene_name
    return {"status": "success"}


@app.get("/agents", response_model=List[AgentInfo])
def get_agents(query: Optional[str] = None):
    agents = simulator.agents

    def fuzzy_search(agents, query):
        return [
            agent
            for agent in agents
            if query.lower() in agent.name.lower() or query == str(agent.id)
        ]

    if query:
        agents = fuzzy_search(agents, query)
    return [
        AgentInfo(
            name=agent.name,
            id=agent.id,
            cls=agent._init_settings["class_name"],
            state=agent.state,
            profile=agent.system_prompt["content"],
        )
        for agent in agents
    ]


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


@app.get("/agent/{id}", response_model=AgentInfo)
def get_agent(id: int):
    agents = simulator.agents
    try:
        agent = agents[id]
        return AgentInfo(
            name=agent.name,
            id=agent.id,
            cls=agent._init_settings["class_name"],
            state=agent.state,
            profile=agent.system_prompt["content"],
        )
    except IndexError:
        return HTMLResponse(content="Agent not found.", status_code=404)


# @app.put("/agent/{id}")
# def put_agent(id: int, new_agent):
#     agents = simulator.agents
#     try:
#         agent = agents[id]
#         agent.update_from_dict(new_agent)
#         return {"status": "success"}
#     except IndexError:
#         return HTMLResponse(content="Agent not found.", status_code=404)


@app.post("/intervention")
def post_intervention(intervention: InterventionMsg):
    global simulator
    map(
        lambda agent: agent.memory.add(
            Msg("assistant", intervention.msg, role="assistant")
        ),
        simulator.agents,
    )
    return {"status": "success"}


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
    return {"status": "success"}


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
    return {"status": "success"}


@app.post("/distributed")
def configure_distributed(req: DistributedConfig):
    global distributed, distributed_args
    simulation_config_path = os.path.join(
        proj_path, "simulation", "examples", _scene, "configs", "simulation_config.yml"
    )
    with open(simulation_config_path, "r") as f:
        simulation_config = yaml.load(f)
    simulation_config["distributed"] = distributed = req.distributed
    with open(simulation_config_path, "w") as f:
        yaml.dump(simulation_config, f)
    distributed_args = req.args
    return {"status": "success"}


@app.get("/messages", response_model=List[MessageUnit])
async def get_messages(
    filter_condition: Optional[FilterCondition] = None,
    offset: Optional[int] = 0,
    limit: Optional[int] = 10,
):
    global cur_msgs, cur_msgs_index
    if cur_msgs is None:
        async with lock:
            cur_msgs = message_manager.messages.copy()
        cur_msgs_index = list(range(len(cur_msgs)))
    msgs = filter_msgs_or_states(cur_msgs, filter_condition)
    return msgs[offset : offset + limit]


@app.get("/messages/random")
async def random_selection_messages(num: int):
    global cur_msgs, cur_msgs_index
    async with lock:
        cur_msgs = message_manager.messages.copy()
    cur_msgs_index = list(range(len(cur_msgs)))
    indexed_cur_msgs = list(enumerate(cur_msgs))
    indexed_chosen_cur_msgs = random.choices(indexed_cur_msgs, k=num)
    cur_msgs_index, cur_msgs = zip(*indexed_chosen_cur_msgs)
    cur_msgs_index = list(cur_msgs_index)
    cur_msgs = list(cur_msgs)
    return {"status": "success"}


@app.get("/states", response_model=List[AgentStateInfo])
async def get_all_agent_states_info():
    module_path = f"simulation.examples.{_scene}.agent"
    all_agent_states_info = importlib.import_module(module_path).ALL_AGENT_STATES
    return [
        AgentStateInfo(agent_cls_name=agent_cls_name, states=states)
        for agent_cls_name, states in all_agent_states_info.items()
    ]


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
async def pause_and_resume():
    if play_event.is_set():
        message_manager.message_queue.put("Pause simulation.")
        play_event.clear()
    else:
        global cur_msgs
        cur_msgs = None
        await message_manager.clear()
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
