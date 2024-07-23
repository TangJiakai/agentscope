# General Simulation
### Pre-download
Create a `Downloads` directory under the `simulation` directory. In this directory, download the embedding model. For the download method, please refer to the link (https://modelscope.cn/models/AI-ModelScope/m3e-base/files).
For example:
1. `git lfs install`
2. `git clone https://www.modelscope.cn/AI-ModelScope/m3e-base.git`

### Pipline
1. Configure the following files in the configs directory of the specific scenario (e.g., `examples/job_seeking`):
    - simulation_config.yml
    - memory_config.json
    - model_configs.json
    - all_xx_agent_configs.json
    - [optional] xx_agent_configs.json
    - all_yy_agent_configs.json
    - [optional] yy_agent_configs.json
    - ......
2. If you set `distributed` to `True` in `simulation_config.yml`, you need to perform the following operation:
Run the script `launch_server.sh <server-num-per-host> <base-port>`
3. Run `simulator.py`