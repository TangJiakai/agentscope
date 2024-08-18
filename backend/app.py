import asyncio
import json
import os
import random
import importlib
import inspect
from pathlib import Path
from queue import Empty, Queue
from threading import Thread, Event
import threading
import time
from typing import Dict, List, Literal, Optional, Union

import aiofiles
from ruamel.yaml import YAML
from agentscope.message import Msg
from contextlib import asynccontextmanager
from fastapi import Body, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from backend.utils.connection import manager
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
    InterventionMsg,
    AgentInfo,
    AgentStateInfo,
    GPTReq,
    ChangedMsg,
)
from simulation.memory import (
    NoneMemory,
    ShortMemory,
    ShortLongMemory,
    ShortLongReflectionMemory,
)
from backend.utils.utils import run_sh, try_serialize_dict
from backend.chatgpt_api import rewritten_responses, rate_responses


yaml = YAML()
proj_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
templates = Jinja2Templates(directory=os.path.join(proj_path, "backend", "templates"))


_scene = "job_seeking"
events: Dict[str, Event] = {}
queue = Queue()
simulator = None
simulation_thread: Thread
lock = threading.Lock()
distributed: bool = True
cur_msgs: List[MessageUnit] = None
backend_server_url: str = None
agent_coordinates: Dict[str, List[float]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Update backend_server_url
    global backend_server_url
    host = os.environ.get("HOST", "0.0.0.0")
    port = os.environ.get("PORT", 9000)
    backend_server_url = f"http://{host}:{port}"
    # Launch LLM
    launch_llm_sh_path = os.path.join(
        proj_path, "llmtuning", "scripts", "launch_llm.sh"
    )
    # run_sh(launch_llm_sh_path)

    yield

    # Kill LLM
    kill_llm_sh_path = os.path.join(proj_path, "llmtuning", "scripts", "kill_llm.sh")
    # run_sh(kill_llm_sh_path)

    # Clean distributed servers
    if distributed:
        kill_server_sh_path = os.path.join(
            proj_path, "simulation", "examples", _scene, "kill_all_server.sh"
        )
        run_sh(kill_server_sh_path)


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
async def websocket_chat_endpoint(websocket: WebSocket, id: str):
    await manager.connect(websocket, id)
    try:
        agent = simulator.get_agent_by_id(id)
        while True:
            data = await websocket.receive_text()
            logger.info(f"Receive chat message: {data}")
            if data == "exit":
                break
            resp = agent.external_interview(data).content
            await manager.send_to_agent(id, resp)
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
    # assign agent coordinates
    for agent in agents:
        if agent.agent_id not in agent_coordinates:
            agent_coordinates[agent.agent_id] = [
                random.uniform(0, 1),
                random.uniform(0, 1),
            ]

    def fuzzy_search(agents, query):
        return [
            agent
            for agent in agents
            if query.lower() in agent.name.lower() or query == agent.agent_id
        ]

    if query:
        agents = fuzzy_search(agents, query)
    return [
        AgentInfo(
            name=agent.name,
            id=agent.agent_id,
            cls=agent._init_settings["class_name"],
            state=agent.get_attr(attr="state"),
            profile=agent.get_attr(attr="_profile"),
            coordinates=agent_coordinates[agent.agent_id],
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
def get_agent(id: str):
    agent = simulator.get_agent_by_id(id)
    if agent:
        return AgentInfo(
            name=agent.name,
            id=agent.agent_id,
            cls=agent._init_settings["class_name"],
            state=agent.get_attr(attr="state"),
            profile=agent.get_attr(attr="_profile"),
            coordinates=agent_coordinates[agent.agent_id],
        )
    else:
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
    agents = simulator.agents
    for agent in agents:
        agent(
            Msg(
                "user",
                None,
                role="user",
                fun="post_intervention",
                params={"intervention": intervention.msg},
            )
        )["content"]
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
    return {"status": "success"}


@app.get("/states", response_model=List[AgentStateInfo])
def get_all_agent_states_info():
    module_path = f"simulation.examples.{_scene}.agent"
    all_agent_states_info = importlib.import_module(module_path).ALL_AGENT_STATES
    return [
        AgentStateInfo(agent_cls_name=agent_cls_name, states=states)
        for agent_cls_name, states in all_agent_states_info.items()
    ]


@app.post("/messages", response_model=List[MessageUnit])
def get_messages_with_filter(
    filter_condition: Optional[FilterCondition] = None,
    offset: Optional[int] = 0,
    limit: Optional[int] = 10,
):
    global cur_msgs
    if cur_msgs is None:
        with lock:
            cur_msgs = message_manager.messages.copy()
    msgs = filter_msgs_or_states(cur_msgs, filter_condition)
    return msgs[offset : offset + limit]


def change_msgs(mode: Literal["rewrite", "rate"], new_msgs: List[ChangedMsg]):
    with lock:
        for new_msg in new_msgs:
            if mode == "rewrite":
                message_manager.messages[new_msg.msg_id].rewritten_response = (
                    new_msg.rewritten_response
                )
            elif mode == "rate":
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


@app.put("/messages/random")
def random_selection_messages(num: int):
    global cur_msgs
    with lock:
        cur_msgs = message_manager.messages.copy()
    cur_msgs = sorted(random.choices(cur_msgs, k=num), key=lambda x: x.msg_id)
    return {"status": "success"}


@app.post("/messages/random/undo")
def undo_random_selection():
    global cur_msgs
    with lock:
        cur_msgs = message_manager.messages.copy()
    return {"status": "success"}


@app.put("/messages/{mode}")
def save_changed_messages(mode: Literal["rewrite", "rate"], msgs: List[ChangedMsg]):
    change_msgs(mode, msgs)
    return {"status": "success"}


@app.post("/gpt")
def chatgpt(req: GPTReq):
    with lock:
        msgs = message_manager.messages.copy()
    msgs = [msgs[id].model_dump(include={"prompt", "completion"}) for id in req.msg_ids]
    if req.mode == "rewrite":
        resps = rewritten_responses(msgs)
        change_msgs(
            "rewrite",
            [
                ChangedMsg(msg_id=msg_id, rewritten_response=resps[idx])
                for idx, msg_id in enumerate(req.msg_ids)
            ],
        )
    elif req.mode == "rate":
        resps = rate_responses(msgs)
        change_msgs(
            "rate",
            [
                ChangedMsg(msg_id=msg_id, rating=resps[idx])
                for idx, msg_id in enumerate(req.msg_ids)
            ],
        )
    return {"status": "success"}


@app.post("/tune/{mode}")
def tune(mode: Literal["rewrite", "rate"]):
    # Tune LLM
    tune_llm_sh_path = os.path.join(proj_path, "llmtuning", "scripts", "tune_llm.sh")
    if mode == "rewrite":
        tuning_mode = "sft"
    elif mode == "rate":
        tuning_mode = "ppo"
    run_sh(tune_llm_sh_path, tuning_mode)
    return {"status": "success"}


@app.post("/export/{mode}")
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
            proj_path, "llmtuning", "datasets", "sft_data", "sft_data.json"
        )
        with open(export_path, "w") as f:
            json.dump(msgs, f, ensure_ascii=False, indent=4)
    elif mode == "rate":
        msgs = [
            {"prompt": msg.prompt, "completion": msg.completion, "rating": msg.rating}
            for msg in msgs
            if msg.rating
        ]
        export_path = os.path.join(
            proj_path, "llmtuning", "datasets", "ppo_data", "ppo_data.json"
        )
        with open(export_path, "w") as f:
            json.dump(msgs, f, ensure_ascii=False, indent=4)
    return {"status": "success"}


@app.post("/api/state")
def post_state(state: StateUnit):
    message_manager.add_state(state)
    return {"status": "success"}


@app.post("/api/message")
def post_messages(message: MessageUnit):
    message_manager.add_message(message)
    return {"status": "success"}


@app.post("/start")
async def start():
    # launch server
    simulation_config_path = os.path.join(
        proj_path, "simulation", "examples", _scene, "configs", "simulation_config.yml"
    )
    with open(simulation_config_path, "r") as f:
        simulation_config = yaml.load(f)
    launch_server_sh_path = os.path.join(
        proj_path, "simulation", "examples", _scene, "launch_server.sh"
    )
    run_sh(
        launch_server_sh_path,
        str(simulation_config["server_num_per_host"]),
        str(simulation_config["base_port"]),
    )
    time.sleep(5)

    module_path = f"simulation.examples.{_scene}.simulator"
    Simulator = importlib.import_module(module_path).Simulator
    global simulator, simulation_thread
    simulator = Simulator()
    agents = simulator.agents
    for agent in agents:
        agent.set_attr("backend_server_url", backend_server_url)
    simulation_thread = Thread(target=simulator.run)
    simulation_thread.start()
    return {"status": "success"}


@app.post("/pause")
async def pause_and_resume():
    if play_event.is_set():
        message_manager.message_queue.put("Pause simulation.")
        play_event.clear()
        while not pause_success_event.is_set():
            await asyncio.sleep(0.1)
        # Distribute MsgID for messages
        with lock:
            for idx, msg in enumerate(message_manager.messages):
                msg.msg_id = idx
    else:
        global cur_msgs
        cur_msgs = None
        message_manager.clear()
        message_manager.message_queue.put("Resume simulation.")
        play_event.set()
        pause_success_event.clear()
    return {"status": "success"}


@app.post("/stop")
async def stop():
    stop_event.set()
    return {"status": "success"}


@app.post("/reset")
async def reset():
    # Clean distributed servers
    if distributed:
        kill_server_sh_path = os.path.join(
            proj_path, "simulation", "examples", _scene, "kill_all_server.sh"
        )
        run_sh(kill_server_sh_path)
    global simulator, simulation_thread, cur_msgs
    simulator = None
    kill_event.set()
    play_event.set()
    simulation_thread.join()
    simulation_thread = None
    cur_msgs = None
    message_manager.clear()
    play_event.clear()
    stop_event.clear()
    kill_event.clear()
    pause_success_event.clear()
    message_manager.message_queue.put("Reset simulation.")
    return {"status": "success"}


@app.get("/", response_class=HTMLResponse)
async def read_root():
    return templates.TemplateResponse("index.html", {"request": {}})


@app.post("/store_message")
async def store_message(save_data_path: str = Body(..., embed=True)):
    data = [x.model_dump() for x in message_manager.messages]
    with open(save_data_path, "w") as f:
        await json.dump(data, f, indent=4)
    message_manager.clear()
    return {"status": "success"}