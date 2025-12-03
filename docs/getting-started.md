# Getting Started

This guide will help you get started with `llm-annotator` and understand its main features.

## Basic Concepts

The `llm-annotator` library is built around the {class}`~llm_annotator.Annotator` class, which provides two main workflows:

1. **Dataset Annotation**: Apply prompts to existing datasets and generate annotations
2. **Dataset Generation**: Create new datasets from scratch using LLM completions

## Your First Annotation

Let's start with a simple sentiment classification task:

```python
from llm_annotator import Annotator

# Define your prompt template
prompt_template = "Classify the sentiment of this review as positive or negative:\n\n{text}\n\nSentiment:"

# Create an annotator (using a context manager for automatic cleanup)
with Annotator(model="meta-llama/Llama-3.2-3B-Instruct", max_model_len=4096) as anno:
    dataset = anno.annotate_dataset(
        output_dir="outputs/sentiment-analysis",
        full_prompt_template=prompt_template,
        dataset_name="stanfordnlp/imdb",
        dataset_split="test",
        max_num_samples=100,
    )
    
    print(f"Annotated {len(dataset)} samples")
    print(dataset[0])  # View the first annotated sample
```

## Using Structured Outputs

You can constrain the model's output to follow a specific JSON schema:

```python
from llm_annotator import Annotator

# Define the output schema
output_schema = {
    "type": "object",
    "properties": {
        "sentiment": {
            "type": "string",
            "enum": ["positive", "negative"]
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1
        }
    },
    "required": ["sentiment", "confidence"]
}

prompt_template = """Analyze the sentiment of this movie review.

Review: {text}

Classification:"""

with Annotator(model="meta-llama/Llama-3.2-3B-Instruct") as anno:
    dataset = anno.annotate_dataset(
        output_dir="outputs/sentiment-structured",
        full_prompt_template=prompt_template,
        dataset_name="stanfordnlp/imdb",
        dataset_split="test",
        max_num_samples=100,
        output_schema=output_schema,  # Enforce structured output
    )
    
    # The output will have 'sentiment' and 'confidence' fields
    print(dataset[0])
```

## Generating New Data

Instead of annotating existing datasets, you can generate entirely new data:

```python
from llm_annotator import Annotator

prompt = "Generate a creative short story about space exploration."

sampling_params = {
    "temperature": 1.0,  # Higher temperature for more creativity
    "top_p": 0.95,
    "max_tokens": 512,
}

with Annotator(model="meta-llama/Llama-3.2-3B-Instruct") as anno:
    dataset = anno.generate_dataset(
        output_dir="outputs/stories",
        prompts=prompt,  # Single prompt
        max_num_samples=50,  # Generate 50 variations
        sampling_params=sampling_params,
    )
```

## Working with Large Datasets

For large datasets, enable streaming to avoid loading everything into memory:

```python
from llm_annotator import Annotator

with Annotator(model="meta-llama/Llama-3.2-3B-Instruct", verbose=True) as anno:
    dataset = anno.annotate_dataset(
        output_dir="outputs/large-annotation",
        full_prompt_template="Translate to French: {text}",
        dataset_name="wmt14",
        dataset_config="fr-en",
        dataset_split="train",
        streaming=True,  # Enable streaming
        max_num_samples=10000,
        cache_input_dataset=True,  # Cache for resumability
    )
```

## Resuming Interrupted Jobs

If a job is interrupted, you can simply rerun the same command. The annotator will:

1. Check the output directory for already-processed samples
2. Skip those samples
3. Continue from where it left off

```python
# Run this command multiple times - it will resume automatically
with Annotator(model="meta-llama/Llama-3.2-3B-Instruct") as anno:
    dataset = anno.annotate_dataset(
        output_dir="outputs/resumable-job",
        full_prompt_template="Summarize: {text}",
        dataset_name="cnn_dailymail",
        dataset_config="3.0.0",
        dataset_split="train",
        max_num_samples=5000,
        # Don't set overwrite=True if you want resumability
    )
```

## Uploading to Hugging Face Hub

You can automatically upload results to the Hugging Face Hub:

```python
from llm_annotator import Annotator

with Annotator(model="meta-llama/Llama-3.2-3B-Instruct") as anno:
    dataset = anno.annotate_dataset(
        output_dir="outputs/hub-upload",
        full_prompt_template="Analyze: {text}",
        dataset_name="stanfordnlp/imdb",
        dataset_split="test",
        max_num_samples=1000,
        new_hub_id="your-username/annotated-imdb",  # Your Hub dataset ID
        upload_every_n_samples=100,  # Upload checkpoint every 100 samples
    )
```

## Custom Validation and Postprocessing

Add custom validation logic and postprocessing:

```python
from llm_annotator import Annotator

def validate_output(sample):
    """Custom validation: ensure response is not empty and has reasonable length"""
    response = sample.get("response", "").strip()
    return len(response) > 10 and len(response) < 1000

def postprocess_output(sample):
    """Custom postprocessing: strip whitespace and lowercase"""
    if "response" in sample:
        sample["response"] = sample["response"].strip()
    return sample

with Annotator(model="meta-llama/Llama-3.2-3B-Instruct") as anno:
    dataset = anno.annotate_dataset(
        output_dir="outputs/validated",
        full_prompt_template="Rewrite in simple English: {text}",
        dataset_name="stanfordnlp/imdb",
        dataset_split="test",
        max_num_samples=100,
        validate_fn=validate_output,
        postprocess_fn=postprocess_output,
        num_retries_invalid=3,  # Retry up to 3 times for invalid outputs
    )
```

## Performance Optimization

### Sorting by Length

Sort your dataset by prompt length to improve batching efficiency:

```python
with Annotator(model="meta-llama/Llama-3.2-3B-Instruct") as anno:
    dataset = anno.annotate_dataset(
        output_dir="outputs/optimized",
        full_prompt_template="Process: {text}",
        dataset_name="stanfordnlp/imdb",
        dataset_split="test",
        sort_by_length=True,  # or "longest_first" or "shortest_first"
    )
```

### Prompt Prefix Caching

Cache common prompt prefixes for faster processing:

```python
prompt_prefix = "You are a helpful assistant. Analyze the following text carefully.\n\n"
prompt_template = prompt_prefix + "Text: {text}\n\nAnalysis:"

with Annotator(model="meta-llama/Llama-3.2-3B-Instruct") as anno:
    dataset = anno.annotate_dataset(
        output_dir="outputs/cached",
        full_prompt_template=prompt_template,
        prompt_template_prefix=prompt_prefix,  # Cache this prefix
        dataset_name="stanfordnlp/imdb",
        dataset_split="test",
    )
```

### Multi-GPU Processing

Use tensor parallelism for larger models:

```python
with Annotator(
    model="meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,  # Use 4 GPUs
    max_num_seqs=128,
    gpu_memory_utilization=0.9,
) as anno:
    dataset = anno.annotate_dataset(
        output_dir="outputs/multi-gpu",
        full_prompt_template="Analyze: {text}",
        dataset_name="stanfordnlp/imdb",
        dataset_split="test",
    )
```

## Next Steps

- Read the {doc}`api-reference` for detailed documentation
- Check out the {doc}`examples` for more use cases
- Visit the [GitHub repository](https://github.com/BramVanroy/llm-annotator) for the source code
