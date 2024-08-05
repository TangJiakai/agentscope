# General Simulation
## Pre-requisites
### Launch Embedding Model
1. [optional] Run 
```python
python simulation/examples/job_seeking/launch_emb_model.py
```
to launch the embedding model server.

2. Get the `embedding_api` (for example, http://localhost:8000/), and fill that URL into `simulation/examples/job_seeking/configs/simulation_config.yml`.

### Config
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
1. 
2. If you set `distributed` to `True` in `simulation_config.yml`, you need to perform the following operation:
Run the script `launch_server.sh <server-num-per-host> <base-port>`
3. Run `simulator.py`