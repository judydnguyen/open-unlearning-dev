# vLLM Chat Server

A high-performance chat server using vLLM for fast inference.

## Installation

Install the required dependencies:

```bash
pip install vllm fastapi uvicorn pydantic
```

## Usage

### Basic Usage

Start the server with a model:

```bash
python servers/chat_server.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --host 0.0.0.0 \
    --port 8000
```

### Advanced Options

```bash
python servers/chat_server.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --dtype bfloat16 \
    --max-model-len 4096
```

### Options

- `--model`: Path to the model (HuggingFace model ID or local path) [required]
- `--host`: Host to bind to (default: 0.0.0.0)
- `--port`: Port to bind to (default: 8000)
- `--tensor-parallel-size`: Number of GPUs for tensor parallelism (default: 1)
- `--gpu-memory-utilization`: Fraction of GPU memory to use (default: 0.9)
- `--trust-remote-code`: Trust remote code (default: False)
- `--dtype`: Data type - auto, float16, bfloat16, float32 (default: auto)
- `--max-model-len`: Maximum model length (default: None)
- `--enable-lora`: Enable LoRA support (default: False)
- `--max-lora-rank`: Maximum LoRA rank (default: 16)

## API Endpoints

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model": "meta-llama/Llama-3.2-1B-Instruct",
  "model_loaded": true
}
```

### Chat (Non-streaming)

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello! How are you?"}
    ],
    "temperature": 0.7,
    "max_tokens": 512
  }'
```

Response:
```json
{
  "message": {
    "role": "assistant",
    "content": "Hello! I'm doing well, thank you for asking..."
  },
  "finish_reason": "stop",
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 25,
    "total_tokens": 40
  }
}
```

### Chat (Streaming)

```bash
curl -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Tell me a story"}
    ],
    "stream": true,
    "temperature": 0.7,
    "max_tokens": 512
  }'
```

### List Models

```bash
curl http://localhost:8000/models
```

## Python Client Example

```python
import requests
import json

# Non-streaming
response = requests.post(
    "http://localhost:8000/chat",
    json={
        "messages": [
            {"role": "user", "content": "What is machine learning?"}
        ],
        "temperature": 0.7,
        "max_tokens": 512
    }
)
print(response.json())

# Streaming
response = requests.post(
    "http://localhost:8000/chat/stream",
    json={
        "messages": [
            {"role": "user", "content": "Tell me a joke"}
        ],
        "stream": True,
        "temperature": 0.7,
        "max_tokens": 512
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        line = line.decode('utf-8')
        if line.startswith('data: '):
            data = line[6:]  # Remove 'data: ' prefix
            if data == '[DONE]':
                break
            try:
                chunk = json.loads(data)
                if 'delta' in chunk:
                    print(chunk['delta']['content'], end='', flush=True)
            except json.JSONDecodeError:
                pass
```

## Request Parameters

- `messages`: List of chat messages with `role` ("user", "assistant", "system") and `content`
- `temperature`: Sampling temperature (0.0-2.0, default: 0.7)
- `top_p`: Top-p sampling parameter (0.0-1.0, default: 0.9)
- `top_k`: Top-k sampling parameter (-1 to disable, default: -1)
- `max_tokens`: Maximum tokens to generate (1-4096, default: 512)
- `stream`: Whether to stream the response (default: false)
- `stop`: Optional list of stop sequences

## Notes

- The server automatically uses the model's chat template if available
- For models without chat templates, a simple formatting is used
- The server supports multi-turn conversations through the messages array
- Streaming responses use Server-Sent Events (SSE) format

