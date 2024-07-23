import json


def try_serialize_dict(data):
    serialized_data = {}
    for key, value in data.items():
        try:
            json.dumps({key: value})
            serialized_data[key] = value
        except (TypeError, ValueError):
            pass
    return serialized_data
