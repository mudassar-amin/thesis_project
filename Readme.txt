
## Dataset

The dataset used for training can be found at the following link: [Piano Dataset on Hugging Face](https://huggingface.co/datasets/mudassar93/data_piano).

To load this dataset into your project, utilize the `datasets` library from Hugging Face. The following code snippet demonstrates how to do so:

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("mudassar93/data_piano")
```

Ensure that you have the `datasets` library installed in your environment. If not, you can install it using pip:

```bash
pip install datasets
```

## Fine-tuned Model

The fine-tuned model, specifically tailored for piano-related tasks, is available here: [llama2-chat-piano on Hugging Face](https://huggingface.co/mudassar93/llama2-chat-piano).

To use this model in your application, you'll need the `transformers` library. Below is the code to load the tokenizer and model:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("mudassar93/llama2-chat-piano")
model = AutoModelForCausalLM.from_pretrained("mudassar93/llama2-chat-piano")
```

If the `transformers` library is not already installed, you can install it using the following pip command:

```bash
pip install transformers
```

This setup allows you to directly integrate our pre-trained model into your projects, enabling a wide range of piano-related natural language understanding and generation tasks.
