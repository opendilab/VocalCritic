#!/usr/bin/env python3
"""
Basic inference script for VocalCritic audio model.

This script demonstrates how to use VocalCritic for music criticism generation.
It follows the pattern from test_qwen25_omni_vllm.py for audio model inference.
"""

import argparse
import os
import sys
from typing import NamedTuple

import librosa
import torch
from vllm import LLM, SamplingParams


class QueryResult(NamedTuple):
    """Container for query inputs and metadata."""
    inputs: dict
    limit_mm_per_prompt: dict[str, int]


def get_multi_audio_query(text: str, audio_path: str) -> QueryResult:
    """
    Prepare a query with audio input for the model.
    
    Args:
        text: The text prompt for the model
        audio_path: Path to the audio file
        
    Returns:
        QueryResult containing formatted inputs for the model
    """
    prompt = (
        f"<|im_start|>system\nYou are a professional music critic with expertise in "
        f"vocal performance, arrangement, and musical analysis. Provide detailed, "
        f"structured appraisals of the audio content.<|im_end|>\n"
        "<|im_start|>user\n<|audio_bos|><|AUDIO|><|audio_eos|>"
        f"{text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    
    # Load audio using librosa
    audio_data, sr = librosa.load(audio_path, sr=None)
    
    return QueryResult(
        inputs={
            "prompt": prompt,
            "multi_modal_data": {
                "audio": [audio_data],
            },
        },
        limit_mm_per_prompt={
            "audio": 1,
        },
    )


@torch.no_grad()
def generate_criticism(model: LLM, audio_path: str, prompt_text: str = None, 
                      sampling_params: SamplingParams = None) -> str:
    """
    Generate music criticism for an audio file.
    
    Args:
        model: Initialized LLM model
        audio_path: Path to the audio file
        prompt_text: Optional custom prompt text (default: standard criticism prompt)
        sampling_params: Optional sampling parameters (default: temperature=0.7, max_tokens=1024)
        
    Returns:
        Generated criticism text
    """
    if prompt_text is None:
        prompt_text = (
            "Please analyze this audio and provide a comprehensive music criticism "
            "covering:\n"
            "1. Vocal performance (technique, expression, control)\n"
            "2. Arrangement and instrumentation\n"
            "3. Harmonic and structural analysis\n"
            "4. Emotional interpretation\n"
            "5. Overall assessment and recommendations"
        )
    
    if sampling_params is None:
        sampling_params = SamplingParams(temperature=0.7, max_tokens=1024)
    
    # Prepare query
    query_result = get_multi_audio_query(prompt_text, audio_path)
    
    # Generate response
    outputs = model.generate(query_result.inputs, sampling_params=sampling_params)
    
    return outputs[0].outputs[0].text


def main():
    parser = argparse.ArgumentParser(
        description="VocalCritic: Music Criticism Generation from Audio"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the VocalCritic model"
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        required=True,
        help="Path to the input audio file"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt text (optional)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate (default: 1024)"
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=8192,
        help="Maximum model length (default: 8192)"
    )
    parser.add_argument(
        "--max_num_seqs",
        type=int,
        default=5,
        help="Maximum number of sequences (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model_path):
        print(f"Error: Model path does not exist: {args.model_path}")
        sys.exit(1)
    
    if not os.path.exists(args.audio_path):
        print(f"Error: Audio file does not exist: {args.audio_path}")
        sys.exit(1)
    
    print(f"Loading model from: {args.model_path}")
    print(f"Processing audio: {args.audio_path}")
    
    try:
        # Initialize model (following pattern from test_qwen25_omni_vllm.py lines 37-45)
        llm = LLM(
            model=args.model_path,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            trust_remote_code=True,
            limit_mm_per_prompt={
                "audio": 1,
            },
        )
        
        sampling_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
        
        # Generate criticism
        print("\nGenerating music criticism...")
        criticism = generate_criticism(
            llm,
            args.audio_path,
            args.prompt,
            sampling_params
        )
        
        print("\n" + "="*80)
        print("MUSIC CRITICISM")
        print("="*80)
        print(criticism)
        print("="*80)
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

