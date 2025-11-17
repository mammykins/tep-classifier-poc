# Sentiment Classification MVP Pipeline

This repository contains a **Minimum Viable Product (MVP)** for implementing automated comment analysis on large volumes of institutional feedback, such as surveys collected from pupils, employees, or other stakeholders. The system uses a zero-shot approach to immediately classify free-text comments (simulated here as student feedback) into actionable categories, like **positive** or **negative** sentiment.  

The data is simulated (made-up comments) for demonstration purposes, but the architecture is designed to be easily adaptable to real-world datasets.  

Credit to the folks at [MantisNLP](https://www.linkedin.com/posts/maupson_were-big-fans-of-hugging-face-at-mantisnlp-activity-7393577817621315586-o25X?utm_source=share&utm_medium=member_desktop&rcm=ACoAABrHWH0BwgvC9d69bW6pcmCksXcb41BCv0g) for inspiration on using zero-shot classification for sentiment analysis.

## Technical Approach

This pipeline is built using modern NLP tooling, emphasizing structured output and low-overhead deployment.

### Core Architecture (Sieves)

The system is built around the **`sieves`** library, which provides a document-based pipeline architecture. The use of `sieves` offers key benefits over direct interaction with raw language models:

1. **Guaranteed Structured Data Output:** The `sieves` pipeline wraps underlying structured generation tools to ensure that the output is consistently valid and structured (e.g., classifying comments strictly as 'positive' or 'negative'). This is essential because zero- or few-shot language models can often be "finicky" or unreliable when prompted directly.
2. **Observability and Debugging:** The document-based structure allows for a clear, step-by-step pipeline, simplifying debugging and tracking the results of the classification process.
3. **Unified Interface:** It allows us to seamlessly integrate models from different libraries (in this case, Hugging Face `transformers`) into a coherent workflow.

### Production Strategy (Hugging Face Inference Endpoints)

While this MVP is run locally on a CPU for immediate demonstration, the strategic choice for future, high-volume production deployment is **Hugging Face Inference Endpoints (HFIE)**.

**HFIE are crucial because they offer a managed deployment solution, eliminating the need for a dedicated DevOps or MLOps team**. Instead of dedicating time to building APIs, managing containers (like AWS ECS), and handling infrastructure, the team can focus purely on applying NLP solutions to drive service improvement. HFIE provides low-latency inference, often running faster than self-managed container services. This approach minimizes the cognitive load for the client organization, making the solution scalable and maintainable.

## Getting Started

This repository uses **`uv`** for fast, reliable dependency management, aligning with strong software engineering principles like **Reproducible Analytical Pipelines (RAP)**.

### Prerequisites

You must have `uv` installed.

### 1. Save the Code

Save the MVP code provided (e.g., the script that defines the `sieves` pipeline and runs the simulated data) into a file named **`sentiment_mvp.py`** in your project root.

### 2. Install Dependencies using `uv`

Use `uv add` to install the required libraries (`sieves` and `transformers`):

```bash
# Adds sieves and transformers to your project environment
uv add sieves transformers gliner2
```

### 3. Run the Pipeline

Execute the script using `uv run`. `uv` ensures the script runs in the correct, managed environment:

```bash
uv run python sentiment_mvp.py
```

### Notes on Authentication

To run this specific MVP using the publicly available zero-shot classification model (`MoritzLaurer/xtremedistil-l6-h256-zeroshot-v1.1-all-33`), **no Hugging Face API key (`HF_TOKEN`) is required** [Conversation History]. The model weights are loaded locally (or accessed publicly), allowing for immediate testing.
