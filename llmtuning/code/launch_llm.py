import argparse
import traceback
from vllm import LLM, SamplingParams
from fastapi import FastAPI, HTTPException, Request
from datetime import datetime
import os
import time


app = FastAPI()

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

# Initialize the LLM with the specified model and parameters
llm = LLM(
    model="/data/pretrain_dir/Meta-Llama-3-8B-Instruct", 
    trust_remote_code=True,
    enable_prefix_caching=True, 
    gpu_memory_utilization=0.8,
    dtype="auto",
    enforce_eager=True,
    tensor_parallel_size=1,
)

cnt = 0
@app.post("/v1/chat/completions")
async def generate_completion(request: Request):
    try:
        global cnt
        cnt+=1
        print("Cnt:",cnt)
        # Read the JSON payload from the request
        system_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"Request received at: {system_time}")
        st=time.time()
        body = await request.json()
        print(f"Received body is : {body}")
        # Extract necessary fields from the JSON
        temperature = body.get("temperature", 1.0)
        max_tokens = body.get("max_tokens", 512)
        #extra_body = body.get("extra_body", {})
        model_name = body.get("model", "")
        messages = body.get("messages", [])

        prompt = "".join([msg['content'] for msg in messages])
        
        # Tokenize the input prompt
        sampling_params = SamplingParams(temperature=temperature, top_p=0.95, max_tokens=max_tokens)
        outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
        response_text = str(outputs[0].outputs[0].text)
        print("[Response]" + response_text)
        # Generate response
        response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "system_fingerprint": "fp_44709d6fcb",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "logprobs": None,  # Or you can replace with an empty dictionary {}
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(outputs[0].prompt_token_ids),
                "completion_tokens": len(outputs[0].outputs[0].token_ids),  # Since response_text is just a single token
                "total_tokens": len(outputs[0].prompt_token_ids) + len(outputs[0].outputs[0].token_ids)
            }
        }
        return response
    except Exception as e:
        error_details = traceback.format_exc()
        raise HTTPException(status_code=500, detail=str(e) + \
                            f"The error prompt is : {prompt}" + \
                            "\nError occurred at:\n" + error_details)


# Run the application with Uvicorn
if __name__ == "__main__":
    import uvicorn
    # get the host and port from the cmd
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--host', type=str, default="localhost", help='host')
    parser.add_argument('--port', type=int, default=8083, help='port')
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)