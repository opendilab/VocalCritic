# VocalCritic

<div align="center">

**VocalCritic: [Paper Title]**

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/XXXX.XXXXX)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

*An advanced multimodal audio model for comprehensive vocal and music criticism*

</div>

## 📖 Overview

VocalCritic is an advanced audio model designed for comprehensive vocal and music criticism. The model leverages multimodal AI capabilities to analyze audio inputs and generate professional, insightful music appraisals.

## 📑 Table of Contents

- [Overview](#-overview)
- [Core Contributions](#-core-contributions)
- [Paper](#-paper)
- [Model Download](#-model-download)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Evaluation](#-evaluation)
- [Requirements](#-requirements)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

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

## 📄 Paper

If you find VocalCritic useful in your research, please cite our paper:

```bibtex
@article{vocalcritic2024,
  title={[Paper Title]},
  author={[Authors]},
  journal={[Journal/Conference]},
  year={2024},
  url={[arXiv/Paper URL]}
}
```

**Paper Link**: [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX) | [PDF](AI_Music_NeurIPS_workshop_final.pdf)

## 📥 Model Download

### Pre-trained Models

Please download the pre-trained VocalCritic models from the following links:

| Model | Size | Download Link | Description |
|-------|------|---------------|-------------|
| VocalCritic-Base | [Size] | [Download Link] | Base model for music criticism |
| VocalCritic-Large | [Size] | [Download Link] | Large model with enhanced capabilities |

**Note**: Please place the downloaded model files in the appropriate directory and update the model path in the inference script.

### Model Weights

- **Hugging Face**: [Link to be added]
- **ModelScope**: [Link to be added]
- **Google Drive**: [Link to be added]

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

## 📊 Evaluation

VocalCritic can be evaluated using the comprehensive benchmark system. See `ReactionEvalBenchmark` for detailed evaluation metrics and usage.

### Evaluation Results

| Metric | Score | Description |
|--------|------|-------------|
| QA Accuracy | [Score]% | Question answering accuracy |
| Completeness | [Score]% | Content completeness score |
| Precision | [Score]% | Factual accuracy |
| Novelty | [Score]% | Creative insights score |
| Overall | [Score]% | Overall performance |

## 📋 Requirements

- **Framework**: vLLM with multimodal audio support
- **Input Format**: Audio files (WAV, MP3, etc.) processed via librosa
- **Output Format**: Structured text appraisals in markdown format
- **GPU**: Recommended NVIDIA GPU with at least [X]GB VRAM
- **Python**: Python 3.8+

## 📄 License

This project is released under the MIT License. See [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

We would like to express our gratitude to the following:

- **Research Community**: Thanks to the open-source community for providing excellent tools and frameworks
- **vLLM Team**: For the excellent vLLM framework that enables efficient inference
- **Qwen Team**: For the Qwen2.5-Omni model architecture and multimodal capabilities
- **Music Information Retrieval Community**: For inspiration and valuable research insights
- **Dataset Contributors**: For providing high-quality music datasets for training and evaluation

### Related Projects

- [ReactionEvalBenchmark](../ReactionEvalBenchmark/): Comprehensive evaluation framework for music appraisal models
- [Qwen2.5-Omni](https://github.com/QwenLM/Qwen2.5-Omni): Multimodal foundation model
- [vLLM](https://github.com/vllm-project/vllm): Fast LLM inference and serving

### Citation

If you use VocalCritic in your research, please cite:

```bibtex
@article{vocalcritic2024,
  title={[Paper Title]},
  author={[Authors]},
  journal={[Journal/Conference]},
  year={2024},
  url={[arXiv/Paper URL]}
}
```

---

**⭐ If you find this project helpful, please consider giving it a star!**

