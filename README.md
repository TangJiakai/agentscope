# General Simulation
## Pre-requisites
### 0. Install Dependencies
**Agentscope**:
Install from source code (https://github.com/pan-x-c/AgentScope/tree/feature/pxc/async_create_agent). 
    
- Modify the default function parameter `timeout` of `call_agent_func` in `src.agentscope.rpc.rpc_client.py` to 60000.
- Add two lines of code:
    ```python
    self.api_key = api_key
    self.client_args = client_args or {}
    ```
In the ``__init__`` function of `OpenAIWrapperBase` in the file `src/agentscope/models/openai_model.py`.
- Modify the variables in the `src/agentscope/constants.py` file:
    ```python
    _DEFAULT_RPC_TIMEOUT = 2
    _DEFAULT_RPC_RETRY_TIMES = 20
    ```

**vllm**
```bash
pip install vllm
```


### 1. Launch Embedding Model
1. Run 
```bash
bash simulation/tools/launch_multi_emb_models.sh
```
to launch the embedding model services.

2. Get the `embedding_api` (for example, [http://localhost:8003/](http://localhost:8003/)), and fill that URL into `simulation/examples/job_seeking/configs/simulation_config.yml`.

### 2. Launch LLM Model
1. Run 
```bash
bash llm/launch_all_llm.sh
```
to launch the LLM model server.

2. Get the `base_url` (for example, [http://localhost:8083/](http://localhost:8083/v1)), and fill that URL into `simulation/examples/job_seeking/configs/model_configs.json`. You can set multiply LLM models.

### 3. Config
Configure the following files in the configs directory of the specific scenario (e.g., `examples/job_seeking`):

    - simulation_config.yml
    - memory_config.json
    - model_configs.json
    - all_xx_agent_configs.json
    - xx_agent_configs.json
    - all_yy_agent_configs.json
    - yy_agent_configs.json
    - ......

p.s. 
1. The `xx` and `yy` in the file names are placeholders for the specific agent types.
2. The `all_xx_agent_configs.json` file is used to store all configurations for xx-type agents (serving the frontend's agent quantity selection operations), while `xx_agent_configs.json` is the configuration file that the simulation will actually read later.

## Pipline
### 1. Launch Distributed Server
Run the following command to launch the distributed server:
```bash
bash simulation/examples/job_seeking/launch_server.sh <server_num_per_host> <base_port>
```

### 2. Run Simulation
Run the following command to run the simulation:
```python
python simulation/examples/job_seeking/simulator.py
```

### 3. Kill Distributed Server
Run the following command to kill the distributed server:
```bash
bash simulation/examples/job_seeking/kill_all_server.sh
```


### Monitor
You can run the command `as_studio` in the terminal to monitor the server.