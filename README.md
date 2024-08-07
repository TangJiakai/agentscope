# General Simulation
## Pre-requisites
### 1. Launch Embedding Model
1. [optional] Run 
```python
python simulation/examples/job_seeking/launch_emb_model.py
```
to launch the embedding model server.

2. Get the `embedding_api` (for example, [http://localhost:8000/](http://localhost:8000/)), and fill that URL into `simulation/examples/job_seeking/configs/simulation_config.yml`.

### 2. Launch LLM Model
1. [optional] Run 
```bash
bash llmtuning/scripts/launch_llm.sh
```
to launch the LLM model server.

2. Get the `base_url` (for example, [http://localhost:8000/](http://localhost:8083/v1)), and fill that URL into `simulation/examples/job_seeking/configs/model_configs.json`.

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


## Tool
### Agentscope Studio
You can run the command `as_studio` in the terminal to monitor the server.