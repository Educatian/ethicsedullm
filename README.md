# AI Ethics Education LLM

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Educatian/ethicsedullm/blob/claude/check-the-p-011CUb2jgGmuME6ZSidadx1s/AI_Ethics_LLM_Training_Colab.ipynb)

A specialized language model fine-tuned for AI ethics education, covering major international frameworks including the EU AI Act, OECD AI Principles, Partnership on AI, and UNESCO Ethical Guidelines.

**🚀 Quick Start:** Click the Colab badge above to train in Google Colab (free GPU available!) • [Colab Guide](COLAB_GUIDE.md)

## Project Overview

This project creates an educational AI assistant that helps students, professionals, and researchers understand complex ethical considerations in AI development and deployment. The model is fine-tuned using QLoRA (Quantized Low-Rank Adaptation) on Meta's Llama 3.1 Instruct, making it efficient while maintaining high-quality responses.

## Key Features

- **Comprehensive Coverage**: Trained on curated content from EU AI Act, OECD, Partnership on AI, UNESCO, and technical ethics literature
- **Efficient Architecture**: QLoRA-based fine-tuning enables training on consumer hardware
- **Rigorous Evaluation**: Hybrid keyword + rubric evaluation system for quality assessment
- **Interactive Interface**: Modern Gradio web UI with adjustable parameters
- **Multiple Frameworks**: Discusses fairness, transparency, accountability, privacy, and sustainability

## Ethical Frameworks Covered

- **EU AI Act**: Risk categories, high-risk systems, compliance requirements
- **OECD AI Principles**: Human-centered values, transparency, accountability
- **Partnership on AI**: Multi-stakeholder collaboration, responsible development
- **UNESCO Ethical Guidelines**: Human rights, environmental considerations, sustainability

## Google Colab (Recommended)

**🎯 Easiest way to get started - No local GPU needed!**

1. Click: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Educatian/ethicsedullm/blob/claude/check-the-p-011CUb2jgGmuME6ZSidadx1s/AI_Ethics_LLM_Training_Colab.ipynb)
2. Select **Runtime → Change runtime type → GPU (T4)**
3. Run cells from top to bottom
4. Enter your Hugging Face token when prompted
5. Wait 2-3 hours for training to complete

**Free T4 GPU is sufficient!** See [COLAB_GUIDE.md](COLAB_GUIDE.md) for detailed instructions.

## Local Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended, 12GB+ VRAM for training)
- 20GB+ free disk space

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/ethicsedullm.git
   cd ethicsedullm
   ```

2. Install dependencies:
   ```bash
   python install_dependencies.py
   # OR
   pip install -r requirements.txt
   ```

3. *(Optional)* Request access to Llama 3.1 models on Hugging Face:
   - Visit: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
   - Accept the license terms
   - Login: `huggingface-cli login`

## Usage

### 1. Data Collection

Prepare comprehensive training data combining curated ethical guidelines and web scraping:

```bash
python data_collection.py
```

This creates `ai_ethics_dataset.jsonl` with:
- 30+ curated Q&A pairs covering major frameworks
- Additional scraped content (if sources are accessible)

### 2. Model Training

Fine-tune Llama 3.1 Instruct using QLoRA:

```bash
python model_training_qlora.py
```

**Training Configuration:**
- Base model: Llama-3.1-8B-Instruct
- Method: QLoRA (4-bit quantization)
- LoRA rank: 16
- Epochs: 3
- Training time: ~2-4 hours on RTX 3090

**Memory Requirements:**
- Training: ~12GB VRAM
- Inference: ~6GB VRAM (with quantization)

### 3. Model Evaluation

Evaluate using hybrid keyword + rubric approach:

```bash
python evaluation_rubric.py
```

**Evaluation Metrics:**
- Accuracy (30%): Factual correctness, framework alignment
- Completeness (25%): Coverage, depth, examples
- Clarity (20%): Structure, readability
- Ethical Awareness (15%): Balanced perspective, trade-offs
- Actionability (10%): Practical guidance

Results saved to `evaluation_report.json`.

### 4. Launch Web Interface

Start the Gradio demo:

```bash
python app.py
```

Access at: http://localhost:7860

**Features:**
- Temperature control for response creativity
- Max token adjustment
- Pre-loaded example questions
- Copy-to-clipboard functionality

### 5. (Optional) Merge LoRA Weights

For deployment without loading adapters separately:

```bash
python merge_lora.py
```

Creates standalone model at `./ai_ethics_llm_merged/`.

## Project Structure

```
ethicsedullm/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── install_dependencies.py            # Automated installation
│
├── prompt_templates.py                # System prompts and formats
├── ethical_guidelines_data.py         # Curated ethical guidelines dataset
├── data_collection.py                 # Data preparation pipeline
│
├── model_training_qlora.py            # QLoRA fine-tuning script
├── merge_lora.py                      # Merge LoRA with base model
│
├── evaluation_rubric.py               # Hybrid evaluation system
├── evaluation.py                      # Legacy evaluation (keyword-only)
├── ethics_test_cases.json             # Test questions with expected themes
│
├── app.py                             # Gradio web interface
└── .gitignore                         # Git ignore patterns
```

## Training Data Details

The dataset (`ethical_guidelines_data.py`) contains 30+ high-quality examples covering:

1. **Regulatory Frameworks**: EU AI Act, GDPR compliance
2. **Ethical Principles**: OECD, UNESCO, Partnership on AI
3. **Technical Concepts**: Fairness metrics, differential privacy, XAI
4. **Practical Implementation**: Model Cards, impact assessments, governance
5. **Domain Applications**: Healthcare, employment, facial recognition
6. **Environmental Ethics**: Carbon footprint, sustainability, green AI

Each example includes:
- Question/instruction
- Detailed response (100-200 words)
- Source framework attribution
- Category classification

## Evaluation Results

Example evaluation metrics (will vary based on training):

```
Average Combined Score:  0.75-0.85
Average Keyword Score:   0.70-0.80
Average Rubric Score:    0.75-0.85

Rubric Breakdown:
  accuracy          : 0.80 (weight: 0.30)
  completeness      : 0.75 (weight: 0.25)
  clarity           : 0.80 (weight: 0.20)
  ethical_awareness : 0.70 (weight: 0.15)
  actionability     : 0.65 (weight: 0.10)
```

## Technical Details

### Model Architecture
- **Base**: Meta Llama 3.1 8B Instruct
- **Fine-tuning**: QLoRA with 4-bit quantization
- **LoRA Config**: rank=16, alpha=32, dropout=0.05
- **Target Modules**: All attention and FFN layers

### Prompt Format
Uses Llama 3.1 Instruct chat template:
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
[System prompt with ethical guidelines]
<|eot_id|><|start_header_id|>user<|end_header_id|>
[User question]
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
[Model response]
```

### Hardware Requirements

**Minimum (Inference only):**
- GPU: 8GB VRAM (e.g., RTX 3060)
- RAM: 16GB
- Storage: 10GB

**Recommended (Training + Inference):**
- GPU: 16GB+ VRAM (e.g., RTX 3090, A100)
- RAM: 32GB
- Storage: 30GB

## Example Questions

- "What are the key principles of the EU AI Act?"
- "Explain different definitions of fairness in machine learning"
- "What ethical considerations arise with facial recognition technology?"
- "How can organizations implement ethical AI governance?"
- "What is differential privacy and why is it important?"
- "How do ethical frameworks address AI's environmental impact?"

## Limitations

- **Educational Tool**: Responses should inform, not replace legal/professional advice
- **Training Data**: Limited to curated examples; may not cover all edge cases
- **Model Size**: 8B parameters; larger models may provide more nuanced responses
- **Temporal**: Based on frameworks as of 2024; may not reflect future regulations
- **Language**: Primarily English; multilingual support limited

## Contributing

Contributions welcome! Areas for improvement:
1. Additional training examples from recent frameworks
2. Multilingual ethical guidelines
3. More sophisticated evaluation rubrics
4. Domain-specific fine-tuning (medical, legal, etc.)
5. Integration with retrieval systems (RAG)

## License

This project is educational and open-source. Please ensure compliance with:
- Llama 3.1 license (https://ai.meta.com/llama/license/)
- Ethical guidelines source attributions

## Citation

If you use this project in research or education, please cite:

```bibtex
@software{ai_ethics_education_llm,
  title={AI Ethics Education LLM},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/ethicsedullm}
}
```

## Resources

- **EU AI Act**: https://artificialintelligenceact.eu/
- **OECD AI Principles**: https://oecd.ai/en/ai-principles
- **Partnership on AI**: https://partnershiponai.org/
- **UNESCO Ethics**: https://www.unesco.org/en/artificial-intelligence
- **Llama 3.1**: https://ai.meta.com/blog/meta-llama-3-1/
- **QLoRA Paper**: https://arxiv.org/abs/2305.14314

## Contact

For questions, issues, or collaborations, please open an issue on GitHub.

---

**Disclaimer**: This AI assistant is designed for educational purposes. Ethical decisions in AI development require human judgment, stakeholder engagement, and consideration of specific contexts. Always consult relevant experts and legal counsel for high-stakes decisions. 
