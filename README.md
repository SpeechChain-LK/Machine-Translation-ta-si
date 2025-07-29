# Tamil-Sinhala Machine Translation Research Project

This repository contains a comprehensive research project focused on developing neural machine translation models for the Tamil-Sinhala language pair. The project explores multiple state-of-the-art transformer architectures and includes complete data preprocessing pipelines, model training experiments, and optimization techniques.

## 🎯 Project Overview

This research investigates machine translation between Tamil and Sinhala languages using various pre-trained transformer models. The project addresses the challenges of low-resource language translation by leveraging transfer learning and fine-tuning approaches.

### Key Features
- Multi-model architecture comparison (T5, mT5, mBART50)
- Comprehensive data preprocessing pipelines
- Multiple training experiments with hyperparameter optimization
- Model quantization for deployment efficiency
- BLEU score evaluation and performance analysis

## 📁 Repository Structure

```
├── 1. Research Papers/              # Literature review and related work
│   ├── sri lanka/                   # Local research papers
│   └── other countries/             # International research papers
│
├── 2. Data Pre-Processing/          # Data cleaning and preparation
│   ├── UCSC Data Pre-Processing/    # University of Colombo datasets
│   ├── UOM Data Pre-Processing/     # University of Moratuwa datasets
│   ├── unwanted charactes/          # Character filtering datasets
│   ├── final-pre-proces.ipynb      # Final preprocessing pipeline
│   └── 80% Train 20% Validation.ipynb # Data splitting
│
├── 3. T5 Traning/                   # Google T5 model experiments
│   ├── T5.ipynb                     # Main T5 training notebook
│   └── EXP 01-03/                   # Experiment results and plots
│
├── 4. MT5 Traning/                  # Multilingual T5 experiments
│   ├── mt5-notebook.ipynb          # mT5 training implementation
│   └── Results/                     # Training outputs and metrics
│
├── 5. MBart50 Tranning/            # Facebook mBART50 experiments
│   ├── mbart50.ipynb               # Main mBART50 notebook
│   └── EXP 01-03/                  # Multiple experiment iterations
│
├── 6. Model Quantization/           # Model optimization
│   └── model quantization.ipynb    # PyTorch quantization implementation
│
└── 7. Testing/                      # Model evaluation
    └── Best Model/                  # Final model artifacts
```

## 🔧 Prerequisites

### Software Requirements
- Python 3.8+
- PyTorch 1.9+
- Transformers library 4.0+
- CUDA-compatible GPU (recommended)

### Required Libraries
```bash
pip install torch torchvision torchaudio
pip install transformers datasets
pip install evaluate sacrebleu bert-score
pip install accelerate
pip install tensorboard
pip install pandas numpy matplotlib seaborn
```

## 📊 Datasets

The project utilizes multiple parallel corpus sources:

### Data Sources
1. **UCSC Dataset**: University of Colombo parallel corpus
2. **UOM Dataset**: University of Moratuwa aligned documents
   - News articles (Army, Hiru, ITN, NewsFirst)
   - Parliamentary order papers
3. **Combined Dataset**: Merged and preprocessed corpus

### Data Statistics
- **Format**: TSV files with `source` (Tamil) and `target` (Sinhala) columns
- **Split Strategies**: Multiple experiments conducted:
  - 80% training, 20% validation
  - 90% training, 10% validation  
  - 80% training, 10% validation, 10% test
- **Preprocessing**: Character filtering, alignment verification, shuffling

## 🚀 Getting Started

### 1. Data Preparation
```bash
# Navigate to data preprocessing directory
cd "2. Data Pre-Processing"

# Run the final preprocessing pipeline
jupyter notebook final-pre-proces.ipynb

# Split data into train/validation sets (multiple strategies tested)
jupyter notebook "80% Train 20% Validation.ipynb"
```

### 2. Model Training

#### T5 Model
```bash
cd "3. T5 Traning"
jupyter notebook T5.ipynb
```

#### mT5 Model
```bash
cd "4. MT5 Traning"
jupyter notebook mt5-notebook.ipynb
```

#### mBART50 Model
```bash
cd "5. MBart50 Tranning"
jupyter notebook mbart50.ipynb
```

### 3. Model Optimization
```bash
cd "6. Model Quantization"
jupyter notebook "model quantization.ipynb"
```

## 🏗️ Model Architectures

### 1. T5 (Text-to-Text Transfer Transformer)
- **Base Model**: `google/t5-small`
- **Approach**: Text-to-text generation
- **Training**: Fine-tuned on Tamil-Sinhala parallel corpus

### 2. mT5 (Multilingual T5)
- **Base Model**: `google/mt5-small`
- **Approach**: Multilingual pre-training
- **Language Codes**: `tam_IN` (Tamil), `sin_LK` (Sinhala)

### 3. mBART50
- **Base Model**: `facebook/mbart-large-50-many-to-many-mmt`
- **Approach**: Multilingual denoising pre-training
- **Language Codes**: `ta_IN` (Tamil), `si_LK` (Sinhala)

## ⚙️ Training Configuration

### Common Hyperparameters
- **Learning Rate**: 1e-4 to 5e-5
- **Batch Size**: 4-16 (depending on model size)
- **Epochs**: 3-5 with early stopping
- **Optimizer**: AdamW
- **Scheduler**: Linear with warmup

### Training Features
- **Early Stopping**: Patience-based stopping
- **Model Checkpointing**: Save best models
- **BLEU Score Monitoring**: Evaluation metric tracking
- **TensorBoard Logging**: Training visualization

## 📈 Evaluation Metrics

The project uses multiple evaluation metrics:

1. **BLEU Score**: Primary translation quality metric
2. **BERTScore**: Semantic similarity evaluation
3. **Loss Tracking**: Training and validation loss monitoring

## 🔧 Model Quantization

The project includes PyTorch dynamic quantization to reduce model size:

- **Technique**: Dynamic quantization with `torch.qint8`
- **Target**: Linear layers quantization
- **Size Reduction**: ~33% reduction (2.4GB → 1.6GB)
- **Performance**: Maintained translation quality

## 📝 Usage Example

```python
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# Load trained model
model = MBartForConditionalGeneration.from_pretrained("./best_model")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

# Translate Tamil to Sinhala
tamil_text = "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?"
inputs = tokenizer(tamil_text, return_tensors="pt", src_lang="ta_IN")

# Generate translation
with torch.no_grad():
    generated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id["si_LK"],
        max_length=128
    )

# Decode output
translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
print(f"Translation: {translation}")
```

## 📊 Experimental Results

The project includes three major experiments for each model with different dataset splitting strategies:

### Dataset Split Experiments
- **Strategy 1**: 80% training, 20% validation
- **Strategy 2**: 90% training, 10% validation
- **Strategy 3**: 80% training, 10% validation, 10% test

### Model Experiments
- **EXP 01**: Baseline training with standard hyperparameters
- **EXP 02**: Hyperparameter optimization and regularization
- **EXP 03**: Advanced training techniques and model improvements

Each experiment folder contains:
- Training notebooks with different data splits
- BLEU score plots and comparisons
- Model checkpoints
- Evaluation results across different split strategies

## 🔍 Research Papers

The `Research Papers` directory contains relevant literature:

### Sri Lankan Research
- Neural Machine Translation for Sinhala-Tamil
- Low-resource NMT approaches
- Statistical Machine Translation studies
- Speech-to-Speech translation research

### International Research
- mBART and multilingual pre-training
- Sequence-to-sequence models
- WMT shared task submissions

## 🤝 Contributing

This is a research project. For contributions or questions:

1. Review the experimental notebooks
2. Check the research papers for theoretical background
3. Examine the data preprocessing pipelines
4. Analyze the training configurations

## 📄 License

This project is for research purposes. Please cite appropriately if using any components.

## 🙏 Acknowledgments

- University of Colombo School of Computing (UCSC) for dataset contributions
- University of Moratuwa (UOM) for parallel corpus data
- Hugging Face Transformers library
- Google Research (T5, mT5)
- Facebook AI Research (mBART)

## 📧 Contact

For research collaboration or questions about this work, please refer to the academic institution associated with this research project.

---

*This README provides an overview of the Tamil-Sinhala Machine Translation research project. For detailed implementation specifics, please refer to the individual notebooks and documentation within each directory.*
