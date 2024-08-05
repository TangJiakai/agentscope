import requests


def get_embedding(sentence, api):
    url = f"{api}/encode"
    response = requests.post(url, json={"sentence": sentence})
    embedding = response.json().get("embedding")
    return embedding


def get_embedding_dimension(api):
    url = f"{api}/embedding-dimension"
    response = requests.get(url)
    embedding_dimension = response.json().get("embedding_dimension")
    return embedding_dimension