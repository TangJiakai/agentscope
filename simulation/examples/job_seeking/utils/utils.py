import json
import re
from loguru import logger


def extract_dict(text):
    pattern = re.compile(r'\{.*?\}', re.DOTALL)
    matches = pattern.findall(text)
    for match in matches:
        json_obj = json.loads(match)
        if json_obj:
            return json_obj
    logger.info("ERROR OUTPUT:\n" + str(text) + "???")
    raise ValueError("ERROR OUTPUT:\n" + str(text) + "???")


def extract_agent_id(agent_id):
    if isinstance(agent_id, str):
        agent_id = agent_id.strip('"\'<>')
    if isinstance(agent_id, str) and agent_id.startswith('[') and agent_id.endswith(']'):
        agent_id = eval(agent_id)
    if isinstance(agent_id, str) and ',' in agent_id:
        agent_id = agent_id.split(',')
    
    if isinstance(agent_id, list):
        res = []
        for aid in agent_id:
            res.append(extract_agent_id(aid))
        return res
    else:
        return agent_id


