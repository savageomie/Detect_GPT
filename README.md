# AI Text Detection: Comparative Study

A comprehensive comparison of three approaches for detecting AI-generated text vs human-written text.

## 🎯 Project Overview

This project implements and compares three state-of-the-art approaches for AI text detection:

1. **Feature-Based ML** - Linguistic features + Classical ML (F1: 0.9197)
2. **Transformer Fine-Tuning** - RoBERTa fine-tuning (F1: 0.9408) ⭐ **Best**
3. **DetectGPT** - Zero-shot probability curvature (F1: 0.6414)

## 📊 Results

| Approach | F1 Score | Accuracy | AUC-ROC |
|----------|----------|----------|---------|
| **Transformer** | **0.9408** | **0.9400** | **0.9931** |
| Feature-Based ML | 0.9197 | 0.9200 | 0.9567 |
| DetectGPT | 0.6414 | 0.6600 | 0.6101 |

## 🚀 Quick Start
```bash
# Setup
git clone <your-repo>
cd ai-text-detection
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Run experiments
python feature_ml_detector.py
python transformer_finetuning.py
python detectgpt_implementation.py

# Compare results
python compare_all_results.py
```

## 📁 Project Structure
```
├── feature_ml_detector.py           # Feature-based approach
├── transformer_finetuning.py        # Transformer approach
├── detectgpt_implementation.py      # DetectGPT approach
├── compare_all_results.py           # Comparison script
├── requirements.txt
└── README.md
```

## 🔮 Next Steps

- [ ] Web UI for real-time detection
- [ ] REST API endpoint
- [ ] Model optimization
- [ ] Multi-language support

## 📝 License

MIT License

## 🙏 Acknowledgments

Dataset: `andythetechnerd03/AI-human-text` (Hugging Face)