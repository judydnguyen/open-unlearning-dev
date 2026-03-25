#!/usr/bin/env python3
"""
Chat server using vLLM for high-performance inference.
Provides REST API endpoints for chat interactions.
"""

import argparse
import asyncio
import json
import logging
import uuid
from typing import List, Optional, Dict, Any, AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

try:
    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    try:
        from vllm.utils import random_uuid
    except ImportError:
        # Fallback if random_uuid is not available
        random_uuid = lambda: str(uuid.uuid4())
except ImportError:
    raise ImportError(
        "vLLM is not installed. Install it with: pip install vllm"
    )

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Request/Response Models
class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message: 'user', 'assistant', or 'system'")
    content: str = Field(..., description="Content of the message")


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="List of chat messages")
    model: Optional[str] = Field(None, description="Model name (if multiple models supported)")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Top-p sampling parameter")
    top_k: int = Field(-1, ge=-1, description="Top-k sampling parameter (-1 for disabled)")
    max_tokens: int = Field(512, ge=1, le=4096, description="Maximum tokens to generate")
    stream: bool = Field(False, description="Whether to stream the response")
    stop: Optional[List[str]] = Field(None, description="Stop sequences")


class ChatResponse(BaseModel):
    message: ChatMessage = Field(..., description="Assistant's response message")
    finish_reason: str = Field(..., description="Reason for finishing: 'stop', 'length', etc.")
    usage: Dict[str, int] = Field(..., description="Token usage statistics")


class HealthResponse(BaseModel):
    status: str
    model: str
    model_loaded: bool


class ChatServer:
    """Chat server using vLLM for inference."""
    
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        trust_remote_code: bool = False,
        dtype: str = "auto",
        max_model_len: Optional[int] = None,
        enable_lora: bool = False,
        max_lora_rank: int = 16,
        **kwargs
    ):
        """
        Initialize the chat server.
        
        Args:
            model_path: Path to the model (HuggingFace model ID or local path)
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory to use
            trust_remote_code: Whether to trust remote code
            dtype: Data type (auto, float16, bfloat16, float32)
            max_model_len: Maximum sequence length
            enable_lora: Enable LoRA support
            max_lora_rank: Maximum LoRA rank
        """
        self.model_path = model_path
        self.engine = None
        self.tokenizer = None
        
        # Store initialization parameters
        self.engine_args = {
            "model": model_path,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "trust_remote_code": trust_remote_code,
            "dtype": dtype,
            "enable_lora": enable_lora,
            "max_lora_rank": max_lora_rank,
            "enforce_eager": True,  # Disable CUDA graph to avoid flash-attn issues
            **kwargs
        }
        
        if max_model_len is not None:
            self.engine_args["max_model_len"] = max_model_len
    
    async def initialize(self):
        """Initialize the vLLM engine asynchronously."""
        logger.info(f"Initializing vLLM engine with model: {self.model_path}")
        
        try:
            # Create async engine args
            engine_args = AsyncEngineArgs(**self.engine_args)
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            
            # Get tokenizer from engine - use get_tokenizer() method
            self.tokenizer = self.engine.get_tokenizer()
            
            logger.info("vLLM engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize vLLM engine: {e}")
            raise
    
    def _format_messages(self, messages: List[ChatMessage]) -> str:
        """
        Format messages for the model using the tokenizer's chat template.
        
        Args:
            messages: List of chat messages
            
        Returns:
            Formatted prompt string
        """
        # Convert to format expected by tokenizer
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        # Use tokenizer's chat template if available
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(
                formatted_messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback: simple formatting
            prompt_parts = []
            for msg in formatted_messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    prompt_parts.append(f"System: {content}\n")
                elif role == "user":
                    prompt_parts.append(f"User: {content}\n")
                elif role == "assistant":
                    prompt_parts.append(f"Assistant: {content}\n")
            prompt_parts.append("Assistant: ")
            prompt = "".join(prompt_parts)
        
        return prompt
    
    async def generate(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = -1,
        max_tokens: int = 512,
        stop: Optional[List[str]] = None,
        stream: bool = False,
    ):
        """
        Generate a response for the given messages.
        
        Args:
            messages: List of chat messages
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            stream: Whether to stream the response
            
        Returns:
            If streaming: AsyncGenerator of text chunks
            If not streaming: Tuple of (generated_text, finish_reason)
        """
        if self.engine is None:
            raise RuntimeError("Engine not initialized. Call initialize() first.")
        
        # Format messages into prompt
        prompt = self._format_messages(messages)
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k if (top_k is not None and top_k > 0) else None,
            max_tokens=max_tokens,
            stop=stop or [],
        )
        
        # Generate response using vLLM async API
        # generate() takes prompt, sampling_params, and request_id directly
        request_id = random_uuid()
        
        if stream:
            # Streaming generation
            async def _stream_generator():
                prev_text = ""
                
                # generate() returns an AsyncGenerator that yields RequestOutput objects
                async for request_output in self.engine.generate(
                    prompt, sampling_params, request_id
                ):
                    if request_output.finished:
                        # Yield any remaining text before finishing
                        if request_output.outputs:
                            for output in request_output.outputs:
                                current_text = output.text if output.text is not None else ""
                                if len(prev_text) < len(current_text):
                                    new_text = current_text[len(prev_text):]
                                    if new_text:
                                        yield new_text
                        break
                    
                    # Get the generated text
                    if request_output.outputs:
                        for output in request_output.outputs:
                            current_text = output.text if output.text is not None else ""
                            # Yield only the new text
                            if len(prev_text) < len(current_text):
                                new_text = current_text[len(prev_text):]
                                if new_text:
                                    yield new_text
                                prev_text = current_text
            
            return _stream_generator()
        else:
            # Non-streaming generation
            generated_text = ""
            finish_reason = "stop"
            
            # generate() returns an AsyncGenerator that yields RequestOutput objects
            async for request_output in self.engine.generate(
                prompt, sampling_params, request_id
            ):
                if request_output.finished:
                    # Get the complete generated text
                    if request_output.outputs:
                        for output in request_output.outputs:
                            generated_text = output.text if output.text is not None else ""
                            finish_reason = output.finish_reason if output.finish_reason is not None else "stop"
                    break
            
            return generated_text, finish_reason


# Global server instance
chat_server: Optional[ChatServer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app."""
    global chat_server
    
    # Startup
    logger.info("Starting chat server...")
    await chat_server.initialize()
    logger.info("Chat server started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down chat server...")
    chat_server = None


# Create FastAPI app
app = FastAPI(
    title="vLLM Chat Server",
    description="High-performance chat server using vLLM",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    global chat_server
    return HealthResponse(
        status="healthy" if chat_server and chat_server.engine else "unhealthy",
        model=chat_server.model_path if chat_server else "unknown",
        model_loaded=chat_server is not None and chat_server.engine is not None,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint for non-streaming responses.
    """
    global chat_server
    
    if chat_server is None or chat_server.engine is None:
        raise HTTPException(status_code=503, detail="Server not ready")
    
    if request.stream:
        raise HTTPException(
            status_code=400,
            detail="Use /chat/stream endpoint for streaming responses"
        )
    
    try:
        # Generate response
        generated_text, finish_reason = await chat_server.generate(
            messages=request.messages,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            max_tokens=request.max_tokens,
            stop=request.stop,
            stream=False,
        )
        
        # Create response
        response_message = ChatMessage(
            role="assistant",
            content=generated_text
        )
        
        # Estimate token usage (approximate)
        try:
            prompt_tokens = len(chat_server.tokenizer.encode(
                chat_server._format_messages(request.messages)
            ))
            completion_tokens = len(chat_server.tokenizer.encode(generated_text))
        except Exception as e:
            logger.warning(f"Failed to estimate token usage: {e}")
            prompt_tokens = 0
            completion_tokens = 0
        
        return ChatResponse(
            message=response_message,
            finish_reason=finish_reason,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
        )
    
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Chat endpoint for streaming responses.
    """
    global chat_server
    
    if chat_server is None or chat_server.engine is None:
        raise HTTPException(status_code=503, detail="Server not ready")
    
    async def generate_stream():
        """Generator for streaming responses."""
        try:
            stream_generator = await chat_server.generate(
                messages=request.messages,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                max_tokens=request.max_tokens,
                stop=request.stop,
                stream=True,
            )
            
            async for text_chunk in stream_generator:
                # Format as Server-Sent Events (SSE)
                data = json.dumps({
                    "delta": {
                        "role": "assistant",
                        "content": text_chunk
                    }
                })
                yield f"data: {data}\n\n"
            
            # Send done signal
            yield "data: [DONE]\n\n"
        
        except Exception as e:
            logger.error(f"Error during streaming: {e}")
            error_data = json.dumps({"error": str(e)})
            yield f"data: {error_data}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/models")
async def list_models():
    """List available models."""
    global chat_server
    if chat_server is None:
        return {"models": []}
    
    return {
        "models": [{
            "id": chat_server.model_path,
            "object": "model",
            "created": 0,
            "owned_by": "open-unlearning"
        }]
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="vLLM Chat Server")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the model (HuggingFace model ID or local path)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Fraction of GPU memory to use"
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Data type"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Maximum model length"
    )
    parser.add_argument(
        "--enable-lora",
        action="store_true",
        help="Enable LoRA support"
    )
    parser.add_argument(
        "--max-lora-rank",
        type=int,
        default=16,
        help="Maximum LoRA rank"
    )
    
    args = parser.parse_args()
    
    # Create chat server
    global chat_server
    chat_server = ChatServer(
        model_path=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        enable_lora=args.enable_lora,
        max_lora_rank=args.max_lora_rank,
    )
    
    # Run server
    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()

