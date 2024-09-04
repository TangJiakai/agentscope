import requests
# import time

# def get_embedding(sentence, api, delay=5):
#     url = f"{api}/encode"
#     attempt = 0
#     while True:
#         attempt += 1
#         try:
#             response = requests.post(url, json={"sentence": sentence}, timeout=7200)
#             response.raise_for_status()  # 检查响应状态码，非200会引发异常
#             embedding = response.json().get("embedding")
#             return embedding
#         except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
#             print(f"Attempt {attempt} to get embedding failed: {e}. Retrying after {delay} seconds...")
#             time.sleep(delay)  # 等待一段时间后重试
#         except requests.exceptions.RequestException as e:
#             raise RuntimeError(f"Request failed with error: {e}, Sentence: {sentence}, URL: {url}")


# def get_embedding_dimension(api, delay=5):
#     url = f"{api}/embedding-dimension"
#     attempt = 0
#     while True:
#         attempt += 1
#         try:
#             response = requests.get(url, timeout=7200)
#             response.raise_for_status()  # 检查响应状态码，非200会引发异常
#             embedding_dimension = response.json().get("embedding_dimension")
#             return embedding_dimension
#         except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
#             print(f"Attempt {attempt} to get embedding dimension failed: {e}. Retrying after {delay} seconds...")
#             time.sleep(delay)  # 等待一段时间后重试
#         except requests.exceptions.RequestException as e:
#             raise RuntimeError(f"Request failed with error: {e}, URL: {url}")

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