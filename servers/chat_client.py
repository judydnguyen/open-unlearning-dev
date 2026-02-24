#!/usr/bin/env python3
"""
Simple console chat client for the vLLM chat server.
Allows interactive chatting via the command line.
"""

import argparse
import json
import requests
import sys
from typing import List, Dict


class ChatClient:
    """Console chat client for the vLLM chat server."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.conversation_history: List[Dict[str, str]] = []
    
    def check_server(self) -> bool:
        """Check if the server is running."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                health = response.json()
                print(f"✓ Server is {health['status']}")
                print(f"  Model: {health['model']}")
                print(f"  Model loaded: {health['model_loaded']}\n")
                return health['model_loaded']
            return False
        except requests.exceptions.RequestException as e:
            print(f"✗ Cannot connect to server at {self.base_url}")
            print(f"  Error: {e}\n")
            return False
    
    def chat(self, message: str, stream: bool = False, **kwargs) -> str:
        """
        Send a chat message to the server.
        
        Args:
            message: User message
            stream: Whether to stream the response
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            Assistant's response
        """
        # Add user message to conversation history
        self.conversation_history.append({"role": "user", "content": message})
        
        # Prepare request
        payload = {
            "messages": self.conversation_history,
            "stream": stream,
            **kwargs
        }
        
        if stream:
            return self._chat_stream(payload)
        else:
            return self._chat_non_stream(payload)
    
    def _chat_non_stream(self, payload: Dict) -> str:
        """Non-streaming chat request."""
        try:
            response = requests.post(
                f"{self.base_url}/chat",
                json=payload,
                timeout=300
            )
            response.raise_for_status()
            result = response.json()
            
            # Extract assistant response
            assistant_message = result['message']['content']
            
            # Add assistant response to conversation history
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })
            
            return assistant_message
            
        except requests.exceptions.RequestException as e:
            return f"Error: {e}"
    
    def _chat_stream(self, payload: Dict) -> str:
        """Streaming chat request."""
        try:
            response = requests.post(
                f"{self.base_url}/chat/stream",
                json=payload,
                stream=True,
                timeout=300
            )
            response.raise_for_status()
            
            full_response = ""
            print("Assistant: ", end="", flush=True)
            
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]  # Remove 'data: ' prefix
                        if data_str == '[DONE]':
                            break
                        try:
                            data = json.loads(data_str)
                            if 'error' in data:
                                return f"Error: {data['error']}"
                            if 'delta' in data and 'content' in data['delta']:
                                chunk = data['delta']['content']
                                print(chunk, end="", flush=True)
                                full_response += chunk
                        except json.JSONDecodeError:
                            pass
            
            print()  # New line after streaming
            self.conversation_history.append({
                "role": "assistant",
                "content": full_response
            })
            
            return full_response
            
        except requests.exceptions.RequestException as e:
            return f"Error: {e}"
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        print("Conversation history cleared.\n")
    
    def show_history(self):
        """Show conversation history."""
        if not self.conversation_history:
            print("No conversation history.\n")
            return
        
        print("\n--- Conversation History ---")
        for i, msg in enumerate(self.conversation_history, 1):
            role = msg['role'].capitalize()
            content = msg['content']
            print(f"\n{i}. {role}:")
            print(f"   {content}")
        print("--- End of History ---\n")
    
    def interactive_mode(self, stream: bool = False, **kwargs):
        """Start interactive chat mode."""
        print("=" * 60)
        print("Interactive Chat Mode")
        print("=" * 60)
        print("Commands:")
        print("  /clear  - Clear conversation history")
        print("  /history - Show conversation history")
        print("  /exit   - Exit the chat")
        print("  /help   - Show this help message")
        print("=" * 60)
        print()
        
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    command = user_input.lower()
                    if command == '/exit' or command == '/quit':
                        print("Goodbye!")
                        break
                    elif command == '/clear':
                        self.clear_history()
                        continue
                    elif command == '/history':
                        self.show_history()
                        continue
                    elif command == '/help':
                        print("\nCommands:")
                        print("  /clear  - Clear conversation history")
                        print("  /history - Show conversation history")
                        print("  /exit   - Exit the chat")
                        print("  /help   - Show this help message\n")
                        continue
                    else:
                        print(f"Unknown command: {user_input}. Type /help for help.\n")
                        continue
                
                # Send message to server
                response = self.chat(user_input, stream=stream, **kwargs)
                
                if not stream:
                    print(f"Assistant: {response}\n")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except EOFError:
                print("\n\nGoodbye!")
                break


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Console chat client for vLLM chat server")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the chat server"
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use streaming responses"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Send a single message and exit (non-interactive)"
    )
    parser.add_argument(
        "message",
        nargs="?",
        help="Message to send (only used with --single)"
    )
    
    args = parser.parse_args()
    
    # Create client
    client = ChatClient(base_url=args.url)
    
    # Check server
    if not client.check_server():
        print("Please make sure the chat server is running.")
        sys.exit(1)
    
    # Prepare generation parameters
    gen_params = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "top_p": args.top_p,
    }
    
    # Single message mode
    if args.single:
        if not args.message:
            print("Error: --single requires a message argument")
            sys.exit(1)
        response = client.chat(args.message, stream=args.stream, **gen_params)
        if not args.stream:
            print(f"\nAssistant: {response}")
        sys.exit(0)
    
    # Interactive mode
    client.interactive_mode(stream=args.stream, **gen_params)


if __name__ == "__main__":
    main()

