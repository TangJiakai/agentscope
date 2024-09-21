import asyncio
import importlib
import inspect
import multiprocessing
import os
import re
import tempfile
import requests
import time
from argparse import Namespace
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import AsyncIterator, Optional, Set
from vllm import ModelRegistry
from fastapi import APIRouter, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Mount
from typing_extensions import assert_never

import vllm.envs as envs
from vllm.config import ModelConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.protocol import AsyncEngineClient
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.cli_args import make_arg_parser
# yapf conflicts with isort for this block
# yapf: disable
from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              ChatCompletionResponse,
                                              CompletionRequest,
                                              CompletionResponse,
                                              DetokenizeRequest,
                                              DetokenizeResponse,
                                              EmbeddingRequest,
                                              EmbeddingResponse, ErrorResponse,
                                              TokenizeRequest,
                                              TokenizeResponse)
# yapf: enable
from vllm.entrypoints.openai.rpc.client import AsyncEngineRPCClient
from vllm.entrypoints.openai.rpc.server import run_rpc_server
#from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_embedding import OpenAIServingEmbedding
from vllm.entrypoints.openai.serving_tokenization import (
    OpenAIServingTokenization)
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext
from vllm.utils import FlexibleArgumentParser, get_open_zmq_ipc_path
from vllm.version import __version__ as VLLM_VERSION
from vllama_judge import LlamaForJudge

import openai
from datetime import datetime
from custom_serving_chat import OpenAIServingChat

TIMEOUT_KEEP_ALIVE = 5  # seconds

async_engine_client: AsyncEngineClient
engine_args: AsyncEngineArgs
openai_serving_chat: OpenAIServingChat
openai_serving_completion: OpenAIServingCompletion
openai_serving_embedding: OpenAIServingEmbedding
openai_serving_tokenization: OpenAIServingTokenization
prometheus_multiproc_dir: tempfile.TemporaryDirectory

# Cannot use __name__ (https://github.com/vllm-project/vllm/pull/4765)
logger = init_logger('vllm.entrypoints.openai.api_server')

_running_tasks: Set[asyncio.Task] = set()


# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
# Register the custom model
ModelRegistry.register_model("LlamaForCausalLM", LlamaForJudge)
token_mapping={'!': '0', '"': '1', '#': '2', '$': '3', '%': '4', '&': '5', "'": '6', '(': '7', ')': '8', '*': '9'}
use_index=False

def model_is_embedding(model_name: str, trust_remote_code: bool,
                       quantization: Optional[str]) -> bool:
    return ModelConfig(model=model_name,
                       tokenizer=model_name,
                       tokenizer_mode="auto",
                       trust_remote_code=trust_remote_code,
                       quantization=quantization,
                       seed=0,
                       dtype="auto").embedding_mode


@asynccontextmanager
async def lifespan(app: FastAPI):

    async def _force_log():
        while True:
            await asyncio.sleep(10)
            await async_engine_client.do_log_stats()

    if not engine_args.disable_log_stats:
        task = asyncio.create_task(_force_log())
        _running_tasks.add(task)
        task.add_done_callback(_running_tasks.remove)

    yield


@asynccontextmanager
async def build_async_engine_client(
        args: Namespace) -> AsyncIterator[Optional[AsyncEngineClient]]:

    # Context manager to handle async_engine_client lifecycle
    # Ensures everything is shutdown and cleaned up on error/exit
    global engine_args
    engine_args = AsyncEngineArgs.from_cli_args(args)

    # Backend itself still global for the silly lil' health handler
    global async_engine_client

    async with build_async_engine_client_from_engine_args(
            engine_args, args.disable_frontend_multiprocessing) as engine:

        async_engine_client = engine  # type: ignore[assignment]
        yield engine


@asynccontextmanager
async def build_async_engine_client_from_engine_args(
    engine_args: AsyncEngineArgs,
    disable_frontend_multiprocessing: bool = False,
) -> AsyncIterator[Optional[AsyncEngineClient]]:
    """
    Create AsyncEngineClient, either:
        - in-process using the AsyncLLMEngine Directly
        - multiprocess using AsyncLLMEngine RPC

    Returns the Client or None if the creation failed.
    """

    # If manually triggered or embedding model, use AsyncLLMEngine in process.
    # TODO: support embedding model via RPC.
    if (model_is_embedding(engine_args.model, engine_args.trust_remote_code,
                           engine_args.quantization)
            or disable_frontend_multiprocessing):
        engine_client = AsyncLLMEngine.from_engine_args(
            engine_args, usage_context=UsageContext.OPENAI_API_SERVER)
        try:
            yield engine_client
        finally:
            engine_client.shutdown_background_loop()
        return

    # Otherwise, use the multiprocessing AsyncLLMEngine.
    else:
        if "PROMETHEUS_MULTIPROC_DIR" not in os.environ:
            # Make TemporaryDirectory for prometheus multiprocessing
            # Note: global TemporaryDirectory will be automatically
            #   cleaned up upon exit.
            global prometheus_multiproc_dir
            prometheus_multiproc_dir = tempfile.TemporaryDirectory()
            os.environ[
                "PROMETHEUS_MULTIPROC_DIR"] = prometheus_multiproc_dir.name
        else:
            logger.warning(
                "Found PROMETHEUS_MULTIPROC_DIR was set by user. "
                "This directory must be wiped between vLLM runs or "
                "you will find inaccurate metrics. Unset the variable "
                "and vLLM will properly handle cleanup.")

        # Select random path for IPC.
        rpc_path = get_open_zmq_ipc_path()
        logger.info("Multiprocessing frontend to use %s for RPC Path.",
                    rpc_path)

        # Build RPCClient, which conforms to AsyncEngineClient Protocol.
        # NOTE: Actually, this is not true yet. We still need to support
        # embedding models via RPC (see TODO above)
        rpc_client = AsyncEngineRPCClient(rpc_path)

        # Start RPCServer in separate process (holds the AsyncLLMEngine).
        context = multiprocessing.get_context("spawn")
        # the current process might have CUDA context,
        # so we need to spawn a new process
        rpc_server_process = context.Process(
            target=run_rpc_server,
            args=(engine_args, UsageContext.OPENAI_API_SERVER, rpc_path))
        rpc_server_process.start()
        logger.info("Started engine process with PID %d",
                    rpc_server_process.pid)

        try:
            while True:
                try:
                    await rpc_client.setup()
                    break
                except TimeoutError:
                    if not rpc_server_process.is_alive():
                        logger.error(
                            "RPCServer process died before responding "
                            "to readiness probe")
                        yield None
                        return

            yield rpc_client  # type: ignore[misc]
        finally:
            # Ensure rpc server process was terminated
            rpc_server_process.terminate()

            # Close all open connections to the backend
            rpc_client.close()

            # Wait for server process to join
            rpc_server_process.join()

            # Lazy import for prometheus multiprocessing.
            # We need to set PROMETHEUS_MULTIPROC_DIR environment variable
            # before prometheus_client is imported.
            # See https://prometheus.github.io/client_python/multiprocess/
            from prometheus_client import multiprocess
            multiprocess.mark_process_dead(rpc_server_process.pid)


router = APIRouter()


def mount_metrics(app: FastAPI):
    # Lazy import for prometheus multiprocessing.
    # We need to set PROMETHEUS_MULTIPROC_DIR environment variable
    # before prometheus_client is imported.
    # See https://prometheus.github.io/client_python/multiprocess/
    from prometheus_client import (CollectorRegistry, make_asgi_app,
                                   multiprocess)

    prometheus_multiproc_dir_path = os.getenv("PROMETHEUS_MULTIPROC_DIR", None)
    if prometheus_multiproc_dir_path is not None:
        logger.info("vLLM to use %s as PROMETHEUS_MULTIPROC_DIR",
                    prometheus_multiproc_dir_path)
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)

        # Add prometheus asgi middleware to route /metrics requests
        metrics_route = Mount("/metrics", make_asgi_app(registry=registry))
    else:
        # Add prometheus asgi middleware to route /metrics requests
        metrics_route = Mount("/metrics", make_asgi_app())

    # Workaround for 307 Redirect for /metrics
    metrics_route.path_regex = re.compile("^/metrics(?P<path>.*)$")
    app.routes.append(metrics_route)


@router.get("/health")
async def health() -> Response:
    """Health check."""
    await async_engine_client.check_health()
    return Response(status_code=200)


@router.post("/tokenize")
async def tokenize(request: TokenizeRequest):
    generator = await openai_serving_tokenization.create_tokenize(request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    elif isinstance(generator, TokenizeResponse):
        return JSONResponse(content=generator.model_dump())

    assert_never(generator)


@router.post("/detokenize")
async def detokenize(request: DetokenizeRequest):
    generator = await openai_serving_tokenization.create_detokenize(request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    elif isinstance(generator, DetokenizeResponse):
        return JSONResponse(content=generator.model_dump())

    assert_never(generator)


@router.get("/v1/models")
async def show_available_models():
    models = await openai_serving_completion.show_available_models()
    return JSONResponse(content=models.model_dump())


@router.get("/version")
async def show_version():
    ver = {"version": VLLM_VERSION}
    return JSONResponse(content=ver)


@router.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest,
                                 raw_request: Request):

    generator = await openai_serving_chat.create_chat_completion(
        request, raw_request,use_index)

    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)

    elif isinstance(generator, ChatCompletionResponse):
        return JSONResponse(content=generator.model_dump())

    return StreamingResponse(content=generator, media_type="text/event-stream")


@router.post("/v1/completions")
async def create_completion(request: CompletionRequest, raw_request: Request):
    generator = await openai_serving_completion.create_completion(
        request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    elif isinstance(generator, CompletionResponse):

        return JSONResponse(content=generator.model_dump())

    return StreamingResponse(content=generator, media_type="text/event-stream")


@router.post("/v1/embeddings")
async def create_embedding(request: EmbeddingRequest, raw_request: Request):
    generator = await openai_serving_embedding.create_embedding(
        request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    elif isinstance(generator, EmbeddingResponse):
        return JSONResponse(content=generator.model_dump())

    assert_never(generator)


if envs.VLLM_TORCH_PROFILER_DIR:
    logger.warning(
        "Torch Profiler is enabled in the API server. This should ONLY be "
        "used for local development!")

    @router.post("/start_profile")
    async def start_profile():
        logger.info("Starting profiler...")
        await async_engine_client.start_profile()
        logger.info("Profiler started.")
        return Response(status_code=200)

    @router.post("/stop_profile")
    async def stop_profile():
        logger.info("Stopping profiler...")
        await async_engine_client.stop_profile()
        logger.info("Profiler stopped.")
        return Response(status_code=200)


def build_app(args: Namespace) -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.include_router(router)
    app.root_path = args.root_path

    mount_metrics(app)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(_, exc):
        err = openai_serving_chat.create_error_response(message=str(exc))
        return JSONResponse(err.model_dump(),
                            status_code=HTTPStatus.BAD_REQUEST)

    if token := envs.VLLM_API_KEY or args.api_key:

        @app.middleware("http")
        async def authentication(request: Request, call_next):
            root_path = "" if args.root_path is None else args.root_path
            if request.method == "OPTIONS":
                return await call_next(request)
            if not request.url.path.startswith(f"{root_path}/v1"):
                return await call_next(request)
            if request.headers.get("Authorization") != "Bearer " + token:
                return JSONResponse(content={"error": "Unauthorized"},
                                    status_code=401)
            return await call_next(request)

    for middleware in args.middleware:
        module_path, object_name = middleware.rsplit(".", 1)
        imported = getattr(importlib.import_module(module_path), object_name)
        if inspect.isclass(imported):
            app.add_middleware(imported)
        elif inspect.iscoroutinefunction(imported):
            app.middleware("http")(imported)
        else:
            raise ValueError(f"Invalid middleware {middleware}. "
                             f"Must be a function or a class.")

    return app


def wait_for_server(url, timeout=100):
    for _ in range(timeout):
        try:
            # 尝试连接到服务器
            response = requests.get(url)
            if response.status_code == 200:
                print("Server is up and running!")
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise TimeoutError(f"Server did not start within {timeout} seconds.")



async def warmup(args,llm_engine):

    print("Warmup prefix caching")
    def extract_macro_instructions(content):
        macro_pattern = r"\{% macro\s+[a-zA-Z_]+_instruction\(\)\s*%\}([\s\S]*?)\{% endmacro %\}"
        macros = re.findall(macro_pattern, content, re.DOTALL)
        return macros
    
    dir_path=args.prompt_dir
    contents=""
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.j2'):
                file_path= os.path.join(root, file)
                with open(file_path, 'r') as f:
                    contents+= f.read()

    instructions = extract_macro_instructions(contents)
    logger.info("Instructions: %s", instructions)
    for instruction in instructions:
        def prepare_request(data, instruction):
            # 修改 content 字段为 instruction
            for message in data['messages']:
                if message['role'] == 'user':
                    message['content'] = instruction

            # 构造 ChatCompletionRequest 实例
            request_data = ChatCompletionRequest(
                model=data['model'],
                messages=data['messages'],
                temperature=data.get('temperature', 0.7),
                max_tokens=data.get('max_tokens', 10),
                stream=data.get('stream', False)
            )

            return request_data
        
        data = {
            'temperature': 0.7,
            'max_tokens': 10,
            'model': '/data/pretrain_dir/Meta-Llama-3-8B-Instruct',
            'messages': [{
                'role': 'user',
                'content': '## Conversation History\nuser: [INSTRUCTION]\n\nYou are a job seeker...'
            }],
            'stream': False
        }

        # 调用函数生成 create_chat_completion 所需的参数
        request_data = prepare_request(data, instruction)
        _ = await openai_serving_chat.create_chat_completion(request_data, None,False)
        blocks = llm_engine.engine.scheduler[0].block_manager.gpu_allocator.evictor.free_table
        for k in blocks.keys():
            blocks[k].last_accessed = datetime(3000, 12, 31, 23, 59, 59).timestamp()
    


async def init_app(
    async_engine_client: AsyncEngineClient,
    args: Namespace,
) -> FastAPI:
    app = build_app(args)

    if args.served_model_name is not None:
        served_model_names = args.served_model_name
    else:
        served_model_names = [args.model]

    model_config = await async_engine_client.get_model_config()

    if args.disable_log_requests:
        request_logger = None
    else:
        request_logger = RequestLogger(max_log_len=args.max_log_len)

    global openai_serving_chat
    global openai_serving_completion
    global openai_serving_embedding
    global openai_serving_tokenization

    openai_serving_chat = OpenAIServingChat(
        async_engine_client,
        model_config,
        served_model_names,
        args.response_role,
        lora_modules=args.lora_modules,
        prompt_adapters=args.prompt_adapters,
        request_logger=request_logger,
        chat_template=args.chat_template,
        return_tokens_as_token_ids=args.return_tokens_as_token_ids,
        enable_auto_tools=args.enable_auto_tool_choice,
        tool_parser=args.tool_call_parser)
    openai_serving_completion = OpenAIServingCompletion(
        async_engine_client,
        model_config,
        served_model_names,
        lora_modules=args.lora_modules,
        prompt_adapters=args.prompt_adapters,
        request_logger=request_logger,
        return_tokens_as_token_ids=args.return_tokens_as_token_ids,
    )
    openai_serving_embedding = OpenAIServingEmbedding(
        async_engine_client,
        model_config,
        served_model_names,
        request_logger=request_logger,
    )
    openai_serving_tokenization = OpenAIServingTokenization(
        async_engine_client,
        model_config,
        served_model_names,
        lora_modules=args.lora_modules,
        request_logger=request_logger,
        chat_template=args.chat_template,
    )
    app.root_path = args.root_path

   
    return app


async def run_server(args, **uvicorn_kwargs) -> None:
    logger.info("vLLM API server version %s", VLLM_VERSION)
    logger.info("args: %s", args)

    async with build_async_engine_client(args) as async_engine_client:
        # If None, creation of the client failed and we exit.
        if async_engine_client is None:
            return

        app = await init_app(async_engine_client, args)
        if args.prompt_dir is not None:
            await warmup(args,async_engine_client)

        shutdown_task = await serve_http(
            app,
            engine=async_engine_client,
            host=args.host,
            port=args.port,
            log_level=args.uvicorn_log_level,
            timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            ssl_ca_certs=args.ssl_ca_certs,
            ssl_cert_reqs=args.ssl_cert_reqs,
            **uvicorn_kwargs,
        )

    # NB: Await server shutdown only after the backend context is exited
    await shutdown_task


if __name__ == "__main__":
    # NOTE(simon):
    # This section should be in sync with vllm/scripts.py for CLI entrypoints.
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    parser.add_argument('--prompt-dir', type=str, help='The file path to the prompt file',default=None)
    parser.add_argument('--use-index', action='store_true', help='Whether to use index')
    args = parser.parse_args()
    if args.use_index:
        use_index=True
    asyncio.run(run_server(args))
