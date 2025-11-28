#!/usr/bin/env python3
"""
Inference script for TASK format model.

Usage:
    # Interactive mode
    python inference.py --model ./checkpoints/final
    
    # Single prompt
    python inference.py --model ./checkpoints/final --prompt "Help me write a Python script"
    
    # With tools
    python inference.py --model ./checkpoints/final --tools
"""

import argparse
import torch
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


def load_model(model_path: str):
    """Load model and tokenizer."""
    print(f"Loading model from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.eval()
    
    print(f"✓ Model loaded on {model.device}")
    return model, tokenizer


def build_prompt(user_message: str, system_prompt: str = None, tools: list = None) -> str:
    """Build a TASK format prompt."""
    parts = []
    
    # System
    if system_prompt:
        parts.append(f'<|im_start|>system')
        parts.append(f'system「{system_prompt}」🏷 sys1')
    else:
        parts.append(f'<|im_start|>system')
        parts.append(f'system「You are a helpful AI assistant that completes tasks step by step.」🏷 sys1')
    
    # Tools
    if tools:
        for tool in tools:
            parts.append(f'tool {{')
            parts.append(f'    name ↦ {tool["name"]} •')
            if tool.get("description"):
                parts.append(f'    description ↦ "{tool["description"]}" •')
            if tool.get("params"):
                parts.append(f'    params ↦ {{')
                for param_name, param_def in tool["params"].items():
                    parts.append(f'        {param_name} ↦ {{ type ↦ {param_def["type"]} }}')
                parts.append(f'    }}')
            parts.append(f'}}')
    
    parts.append('<|im_end|>')
    parts.append('')
    
    # User
    parts.append('<|im_start|>user')
    parts.append(f'user「{user_message}」🏷 usr1')
    parts.append('<|im_end|>')
    parts.append('')
    
    # Start assistant turn
    parts.append('<|im_start|>assistant')
    
    return '\n'.join(parts)


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 4096, temperature: float = 0.7, stream: bool = True) -> str:
    """Generate a response with optional streaming."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Get stop token IDs - both EOS and <|im_end|>
    stop_token_ids = [tokenizer.eos_token_id]
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id != tokenizer.unk_token_id:
        stop_token_ids.append(im_end_id)
    
    if stream:
        # Streaming generation
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
        
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=stop_token_ids,
            streamer=streamer,
        )
        
        # Run generation in a thread
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Stream output
        generated_text = ""
        for new_text in streamer:
            # Stop if we hit end token
            if "<|im_end|>" in new_text:
                new_text = new_text.split("<|im_end|}")[0]
                print(new_text, end="", flush=True)
                generated_text += new_text
                break
            print(new_text, end="", flush=True)
            generated_text += new_text
        
        thread.join()
        print()  # Newline at end
        return generated_text.strip()
    else:
        # Non-streaming generation
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=stop_token_ids,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Extract just the assistant response
        if '<|im_start|>assistant' in response:
            response = response.split('<|im_start|>assistant')[-1]
        if '<|im_end|>' in response:
            response = response.split('<|im_end|>')[0]
        
        return response.strip()


def interactive_mode(model, tokenizer, use_tools: bool = False):
    """Interactive chat mode."""
    print("\n" + "="*60)
    print("TASK Model Interactive Mode")
    print("Type 'quit' to exit, 'clear' to reset")
    print("="*60 + "\n")
    
    # Example tools
    tools = None
    if use_tools:
        tools = [
            {
                "name": "search_web",
                "description": "Search the web for information",
                "params": {"query": {"type": "string"}}
            },
            {
                "name": "run_code",
                "description": "Execute Python code",
                "params": {"code": {"type": "string"}}
            },
        ]
        print("Tools enabled: search_web, run_code\n")
    
    while True:
        try:
            user_input = input("\033[94mYou:\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        if user_input.lower() == 'clear':
            print("\n" + "="*60 + "\n")
            continue
        
        prompt = build_prompt(user_input, tools=tools)
        
        print("\033[92mAssistant:\033[0m")
        response = generate(model, tokenizer, prompt, stream=True)
        print()  # Extra newline after streaming


def main():
    parser = argparse.ArgumentParser(description="TASK model inference")
    parser.add_argument("--model", type=str, default="./checkpoints/final", help="Model path")
    parser.add_argument("--prompt", type=str, help="Single prompt (non-interactive)")
    parser.add_argument("--system", type=str, help="System prompt")
    parser.add_argument("--tools", action="store_true", help="Enable example tools")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    args = parser.parse_args()
    
    model, tokenizer = load_model(args.model)
    
    if args.prompt:
        # Single prompt mode
        tools = None
        if args.tools:
            tools = [
                {"name": "search_web", "description": "Search the web", "params": {"query": {"type": "string"}}},
                {"name": "run_code", "description": "Execute Python code", "params": {"code": {"type": "string"}}},
            ]
        
        prompt = build_prompt(args.prompt, system_prompt=args.system, tools=tools)
        print("\n--- Prompt ---")
        print(prompt)
        print("\n--- Response ---")
        response = generate(model, tokenizer, prompt, args.max_tokens, args.temperature, stream=not args.no_stream)
        if args.no_stream:
            print(response)
    else:
        # Interactive mode
        interactive_mode(model, tokenizer, use_tools=args.tools)


if __name__ == "__main__":
    main()

