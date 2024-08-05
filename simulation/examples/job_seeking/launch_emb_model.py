import argparse
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer
import torch
import uvicorn

def create_app(model_path: str, gpu: int):
    # Initialize the SentenceTransformer model
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    emb_model = SentenceTransformer(model_path, device=device)

    # Define a FastAPI app
    app = FastAPI()

    # Define a Pydantic model for the request body
    class SentenceRequest(BaseModel):
        sentence: str

    # Define the route for encoding sentences
    @app.post("/encode")
    async def encode_sentence(request: SentenceRequest):
        # 从请求中提取单个句子
        sentence = request.sentence
        # 获取句子的嵌入
        embedding = emb_model.encode(sentence, normalize_embeddings=True)
        # 将嵌入转换为列表以便 JSON 序列化
        embedding_list = embedding.tolist()
        return {"embedding": embedding_list}
    
    @app.get("/embedding-dimension")
    async def get_sentence_embedding_dimension():
        # Get the embedding dimension from the model
        embedding_dimension = emb_model.get_sentence_embedding_dimension()
        return {"embedding_dimension": embedding_dimension}

    # You can also define a health check route
    @app.get("/")
    async def read_root():
        return {"message": "Sentence embedding API is up and running!"}

    return app

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run a FastAPI server for sentence embedding.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use (default: 0)")
    parser.add_argument("--model_path", type=str, default="/data/tangjiakai/general_simulation/simulation/Downloads/m3e-base", help="Path to the model")
    parser.add_argument("--port", type=int, default=8000, help="Port number to run the server on (default: 8000)")

    args = parser.parse_args()

    # Create FastAPI app
    app = create_app(args.model_path, args.gpu)

    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=args.port)

if __name__ == "__main__":
    main()