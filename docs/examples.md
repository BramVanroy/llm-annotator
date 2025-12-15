# Examples

This page contains practical examples demonstrating various use cases of the `llm-annotator` library.

## Sentiment Classification

Classify movie reviews as positive or negative using structured output.

```python
from llm_annotator import Annotator

# Define prompt and schema
prompt_prefix = """Analyze the sentiment of the following movie review.

Review: 
"""
prompt_template = prompt_prefix + """{text}

Classification:"""

output_schema = {
    "type": "object",
    "properties": {
        "sentiment": {"type": "string", "enum": ["positive", "negative"]}
    },
    "required": ["sentiment"],
}

# Annotate the dataset
with Annotator(
    model="meta-llama/Llama-3.2-3B-Instruct",
    max_model_len=4096,
    verbose=True
) as anno:
    ds = anno.annotate_dataset(
        output_dir="outputs/sentiment-imdb",
        prompt_template=prompt_template,
        dataset_name="stanfordnlp/imdb",
        dataset_split="test",
        max_num_samples=1000,
        output_schema=output_schema,
        keep_columns=["text", "label"],
        sort_by_length=True,
    )
    
    print(f"Processed {len(ds)} samples")
```

## Named Entity Recognition (NER) Data Generation

Generate synthetic NER training data from scratch with a system message.

```python
from llm_annotator import Annotator

# Define the task prompt
system_message = """You are a data generator for Named Entity Recognition (NER) tasks.
Generate diverse and realistic sentences containing named entities.

Instructions:
- Create natural sentences with 2-5 named entities
- Include person names (PER), locations (LOC), and organizations (ORG)
- Vary sentence complexity and domain

"""

prompt = "Generate a new sentence:"

# Define structured output schema
output_schema = {
    "type": "object",
    "properties": {
        "sentence": {"type": "string"},
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "label": {"type": "string", "enum": ["PER", "LOC", "ORG"]},
                    "start": {"type": "integer"},
                    "end": {"type": "integer"}
                },
                "required": ["text", "label"]
            }
        }
    },
    "required": ["sentence", "entities"]
}

sampling_params = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 64,
    "max_tokens": 256,
}

with Annotator(
    model="meta-llama/Llama-3.2-3B-Instruct",
    max_model_len=2048,
    verbose=True
) as anno:
    ds = anno.generate_dataset(
        output_dir="outputs/ner-generated",
        prompts=prompt,
        system_message=system_message,
        max_num_samples=1000,
        output_schema=output_schema,
        sampling_params=sampling_params,
    )

print(f"Generated {len(ds)} NER samples")
```

## Translation Task

Translate a dataset to another language with quality validation.

```python
from llm_annotator import Annotator

# Custom validation to ensure translations are not empty
def validate_translation(sample):
    translation = sample.get("response", "").strip()
    # Check that translation exists and is not just the original
    return len(translation) > 0 and translation != sample.get("source_text", "")

# Postprocessing to clean up output
def postprocess_translation(sample):
    if "response" in sample:
        # Remove extra whitespace
        sample["response"] = " ".join(sample["response"].split())
    return sample

prompt_template = """Translate the following English text to French. Provide only the translation, without any explanations.

English: {text}

French:"""

with Annotator(
    model="meta-llama/Llama-3.2-3B-Instruct",
    verbose=True
) as anno:
    ds = anno.annotate_dataset(
        output_dir="outputs/translation-en-fr",
        prompt_template=prompt_template,
        dataset_name="wmt14",
        dataset_config="fr-en",
        dataset_split="test",
        max_num_samples=5000,
        keep_columns=["translation"],
        validate_fn=validate_translation,
        postprocess_fn=postprocess_translation,
        num_retries_invalid=3,
    )
```

## Large-Scale Data Annotation with Hub Upload

Process a large dataset with streaming and automatic Hub uploads.

```python
from llm_annotator import Annotator

prompt_template = """Summarize the following article in 2-3 sentences.

Article: {article}

Summary:"""

with Annotator(
    model="meta-llama/Llama-3.2-8B-Instruct",
    tensor_parallel_size=2,  # Use 2 GPUs
    verbose=True
) as anno:
    ds = anno.annotate_dataset(
        output_dir="outputs/summaries-cnn",
        prompt_template=prompt_template,
        dataset_name="cnn_dailymail",
        dataset_config="3.0.0",
        dataset_split="train",
        streaming=True,
        max_num_samples=50000,
        new_hub_id="your-username/cnn-summaries",
        upload_every_n_samples=5000,  # Upload every 5K samples as backup
        cache_input_dataset=True,
        sort_by_length=True,
    )
```

## Multi-Turn Dialogue Generation

Generate conversations for chatbot training.

```python
from llm_annotator import Annotator

prompt_template = """Generate a natural dialogue between a customer and a helpful customer service agent about {topic}.

Requirements:
- Include 3-4 turns per person
- Keep responses professional and helpful
- Address the customer's concern completely

Dialogue:"""

topics = [
    "returning a defective product",
    "tracking a late delivery",
    "changing account information",
    "requesting a refund",
    "technical support for a device",
]

# Create prompts for each topic
prompts = [prompt_template.format(topic=topic) for topic in topics]

sampling_params = {
    "temperature": 0.9,
    "top_p": 0.95,
    "max_tokens": 512,
}

with Annotator(
    model="meta-llama/Llama-3.2-3B-Instruct",
    verbose=True
) as anno:
    ds = anno.generate_dataset(
        output_dir="outputs/dialogues",
        prompts=prompts,  # Each topic gets one sample
        sampling_params=sampling_params,
    )
```

## Data Augmentation

Augment an existing dataset by paraphrasing.

```python
from llm_annotator import Annotator

prompt_template = """Paraphrase the following text while preserving its meaning and sentiment.
Provide only the paraphrased version without any explanations.

Original: {text}

Paraphrased:"""

def validate_paraphrase(sample):
    paraphrase = sample.get("response", "").strip()
    original = sample.get("text", "")
    # Ensure paraphrase is different but has similar length
    return (
        len(paraphrase) > 0 
        and paraphrase != original
        and 0.5 <= len(paraphrase) / len(original) <= 2.0
    )

with Annotator(
    model="meta-llama/Llama-3.2-3B-Instruct",
    verbose=True
) as anno:
    ds = anno.annotate_dataset(
        output_dir="outputs/paraphrased-imdb",
        prompt_template=prompt_template,
        dataset_name="stanfordnlp/imdb",
        dataset_split="train",
        max_num_samples=10000,
        keep_columns=["text", "label"],
        validate_fn=validate_paraphrase,
        num_retries_invalid=2,
        sampling_params={"temperature": 0.7, "top_p": 0.9},
    )
```

## Question-Answer Pair Generation

Generate Q&A pairs from documents.

```python
from llm_annotator import Annotator

prompt_template = """Based on the following context, generate a question and answer pair.
The question should be answerable from the context.

Context: {text}

Generate Q&A pair:"""

output_schema = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "answer": {"type": "string"},
        "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]}
    },
    "required": ["question", "answer"]
}

def validate_qa(sample):
    question = sample.get("question", "").strip()
    answer = sample.get("answer", "").strip()
    # Ensure both question and answer are present and reasonable length
    return (
        len(question) > 10 
        and len(answer) > 5
        and question.endswith("?")
    )

with Annotator(
    model="meta-llama/Llama-3.2-3B-Instruct",
    verbose=True
) as anno:
    ds = anno.annotate_dataset(
        output_dir="outputs/qa-pairs",
        prompt_template=prompt_template,
        dataset_name="wikipedia",
        dataset_config="20220301.en",
        dataset_split="train",
        max_num_samples=5000,
        output_schema=output_schema,
        validate_fn=validate_qa,
        keep_columns=["text"],
    )
```

## Code Documentation Generation

Generate documentation for code snippets.

```python
from llm_annotator import Annotator

prompt_template = """Analyze the following Python function and generate comprehensive documentation.

Function:
```python
{code}
```

Generate documentation:"""

output_schema = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "parameters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "description": {"type": "string"}
                },
                "required": ["name", "description"]
            }
        },
        "returns": {"type": "string"},
        "examples": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["summary", "returns"]
}

with Annotator(
    model="meta-llama/Llama-3.2-3B-Instruct",
    max_model_len=4096,
    verbose=True
) as anno:
    ds = anno.annotate_dataset(
        output_dir="outputs/code-docs",
        prompt_template=prompt_template,
        dataset_name="code_search_net",
        dataset_config="python",
        dataset_split="train",
        max_num_samples=1000,
        output_schema=output_schema,
        keep_columns=["code", "func_name"],
    )
```

## See Also

- {doc}`getting-started` - Learn the basics
- {doc}`api-reference` - Detailed API documentation
- [GitHub Examples](https://github.com/BramVanroy/llm-annotator/tree/main/examples) - Full working examples
