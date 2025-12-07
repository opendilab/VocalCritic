# VocalCritic

<div align="center">

**VocalCritic: Generative Multi-modal Feedback for Singing Voice Synthesis Evaluation**

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2512.02523) | [![Model](https://img.shields.io/badge/License-MIT-blue)](https://huggingface.co/xiaoyi1734/Multimodal-Reaction-Model)

🚀 **Accepted to NeurIPS 2025 AI for Music**

*An advanced multimodal audio model for comprehensive vocal and music criticism*

</div>

## 📖 Overview

VocalCritic is an advanced audio model designed for comprehensive vocal and music criticism. The model leverages multimodal AI capabilities to analyze audio inputs and generate professional, insightful music appraisals.

## 📑 Table of Contents

- [Overview](#-overview)
- [Core Contributions](#-core-contributions)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Citation](#-citation)

## 🎯 Core Contributions

### Key Features

1. **Multimodal Audio Understanding**: VocalCritic processes raw audio inputs to extract nuanced musical elements including vocal techniques, instrumentation, arrangement, and emotional expression.

2. **Professional Music Criticism**: The model generates detailed, structured music appraisals that cover:
   - Vocal performance analysis (technique, expression, control)
   - Arrangement and instrumentation evaluation
   - Harmonic and structural analysis
   - Emotional interpretation and cultural context
   - Comparative analysis with reference works

3. **Comprehensive Evaluation Framework**: Integrated with a multi-dimensional evaluation system that assesses:
   - Factual accuracy and knowledge
   - Content completeness
   - Precision in technical analysis
   - Novelty and creative insights

### Important Results

- **High-Quality Analysis**: The model demonstrates strong performance in generating professional-grade music criticism that balances technical expertise with accessible explanations
- **Multimodal Capability**: Successfully processes audio inputs directly, eliminating the need for separate transcription or feature extraction steps
- **Structured Output**: Produces well-organized appraisals with clear sections covering different aspects of musical analysis
- **Evaluation Performance**: Achieves strong scores across multiple evaluation dimensions including completeness, precision, and novelty

## 🔧 Installation

```bash
pip install vllm librosa torch
```

## 🚀 Quick Start

### Basic Inference

The following script demonstrates how to use VocalCritic for basic audio inference:

```python
from vllm import LLM, SamplingParams
import librosa
from typing import NamedTuple

# Initialize the model
llm = LLM(
    model="your_model_path",  # Path to VocalCritic model
    max_model_len=8192,
    max_num_seqs=5,
    trust_remote_code=True,
    limit_mm_per_prompt={
        "audio": 1,
    },
)

sampling_params = SamplingParams(temperature=0.7, max_tokens=1024)

# Prepare audio input
audio_path = "path/to/your/audio.wav"
audio_data, sr = librosa.load(audio_path, sr=None)

# Create prompt for music criticism
prompt = """<|im_start|>system
You are a professional music critic with expertise in vocal performance, arrangement, and musical analysis. Provide detailed, structured appraisals of the audio content.<|im_end|>
<|im_start|>user
<|audio_bos|><|AUDIO|><|audio_eos|>
Please analyze this audio and provide a comprehensive music criticism covering vocal performance, arrangement, instrumentation, and emotional expression.<|im_end|>
<|im_start|>assistant
"""

# Prepare inputs
inputs = {
    "prompt": prompt,
    "multi_modal_data": {
        "audio": [audio_data],
    },
}

# Generate response
outputs = llm.generate(inputs, sampling_params=sampling_params)
criticism = outputs[0].outputs[0].text

print(criticism)
```

### Using the Inference Script

For a complete example, see `inference.py`:

```bash
python inference.py --model_path /path/to/vocalcritic/model --audio_path /path/to/audio.wav
```

### Citation

If you use VocalCritic in your research, please cite:

```bibtex
@misc{li2025generative,
      title={Generative Multi-modal Feedback for Singing Voice Synthesis Evaluation}, 
      author={Xueyan Li and Yuxin Wang and Mengjie Jiang and Qingzi Zhu and Jing Zhang and Zoey Kim and Yazhe Niu},
      year={2025},
      eprint={2512.02523},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2512.02523}, 
}
```


