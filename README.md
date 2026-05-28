# LLM From Scratch

Building the core components of a Large Language Model from the ground up — implementing each building block to understand how LLMs actually work under the hood.

## Motivation

This repository is a hands-on exploration of LLM internals. The goal is to implement each component from scratch to build a deep, first-principles understanding of how language models are constructed and trained.

## Goals

- Break an LLM down into its fundamental building blocks
- Implement each component step by step
- Understand the *why* behind each design choice, not just the *how*

## Roadmap

- [ ] Tokenizer (BPE / byte-level encoding)
- [ ] Token & positional embeddings
- [ ] Self-attention mechanism
- [ ] Multi-head attention
- [ ] Transformer block (attention + feedforward + layer norm + residuals)
- [ ] Full GPT-style architecture
- [ ] Training loop & loss computation
- [ ] Text generation / sampling (greedy, top-k, top-p, temperature)
- [ ] Fine-tuning experiments

## Project Structure

```
llm-from-scratch/
├── src/            # Core implementation modules
├── notebooks/      # Experiments and explanations
├── data/           # Training data
└── README.md
```

## Getting Started

```bash
git clone https://github.com/<your-username>/llm-from-scratch.git
cd llm-from-scratch
pip install -r requirements.txt
```

## Tech Stack

- Python
- PyTorch / NumPy

## References

- *Attention Is All You Need* (Vaswani et al., 2017)
- Andrej Karpathy's "Let's build GPT" series
- Sebastian Raschka's *Build a Large Language Model (From Scratch)*

## License

MIT