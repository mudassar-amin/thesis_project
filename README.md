
## Abstract
This thesis explores the effectiveness of Large Language Models (LLMs) in enhancing educational methodologies, particularly focusing on personalized learning experiences in music education. Initially, a comprehensive literature review was conducted to establish the theoretical foundation and identify gaps in the current application of Large Language Models in education. Subsequently, employing a quantitative approach, the study utilized the Supervised Fine-Tuning QLoRA approach to adapt the Llama2-chat model to respond accurately to music educational queries. The Results showed the fine-tuned model with the instruction dataset provides some good results on the provided prompts. The performance of the model was evaluated using standard metrics such as BERTScore, F1 Score, and Exact_Match, which confirmed the model's efficacy in providing accurate and contextually appropriate responses. While the findings confirm the potential of integrating LLMs into educational frameworks, they also highlight some limitations, such as the need for continuous model training to adapt to evolving and diverse musical content and creativity. This study establishes a basis for future research, suggesting the exploration of symbolic music understanding models like MUsicBERT and LLMs integration within music education.

## Introduction
In recent years, the capabilities of artificial intelligence (AI) have expanded significantly, particularly in the field of natural language processing (NLP). Large Language Models (LLMs), a sophisticated form of artificial intelligence, can generate and understand human language. This thesis explores the applications of LLMs in music education, specifically within music education. These models work by using a huge collection of written text to learn how to generate and understand human language.


Music education is challenged by different issues like limited access to quality instruction due to geographical and financial barriers, which lack students' ability to get high-quality music training. The diversity in students' learning styles and rates of progress complicates the effectiveness of traditional teaching methods.  Traditional music education methods, which often depend on one-on-one lessons or small classes, are not easily scalable, limiting the availability of quality education. 

The study showed that Large Language Models (LLMs) enhance education by offering personalized learning experiences, serving as support tools, providing assessments and feedback on student work, and generating a variety of educational resources and content to enrich learning/teaching materials.  

As we know, Large Language Models (LLMs) can automate tasks, but they also have some limitations such as biased output and hallucinations. To solve these issues for downstream tasks such as music education, it is necessary to control the model's response using fine-tuning method. Fine-tuning is the technique to increase the performance of LLMs in music education, where models are trained on music datasets in a supervised learning manner. This fine-tuning allows LLMs to understand and generate music content, therefore increasing their accuracy and performance in music education. Moreover, fine-tuning these models requires a lot of computational resources and VRAM due to their larger size, so we used parameter-efficient fine-tuning like QLoRA. QLoRA backpropagates gradients through a frozen, 4-bit quantized pre-trained language model into Low-Rank Adapters (LoRA). 

Through this approach, we were able to fine-tune the Llama2 7B model with limited memory and resources. The Results showed the fine-tuned model with the instruction dataset provides some great results on the provided prompts. The performance of the model was evaluated using standard metrics such as BERTScore, F1
Score, and Exact-Match, which confirmed the model’s efficacy in providing accurate and contextually appropriate responses. While the findings confirm the potential of integrating LLMs into educational frameworks, they also highlight some limitations, such as the need for continuous model training to adapt to evolving and diverse musical content and creativity. This study establishes a basis for future research, suggesting
the exploration of symbolic music understanding models like MusicBERT and LLMs integration within music education.



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

The fine-tuned model, is available here: [llama2-chat-piano on Hugging Face](https://huggingface.co/mudassar93/llama2-chat-piano).

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

# Fine-Tuning Llama 2 with LoRA and QLoRA

This README summarizes the process of fine-tuning the Llama 2 model for a music-related task, leveraging Low-Rank Adaptation (LoRA) and Quantized Low-Rank Adaptation (QLoRA) for efficiency.

## Process Overview

1. **Library Installation**: Install all necessary libraries, including `transformers`, `datasets`, `accelerate`, `peft`, `trl`, and `bitsandbytes`.

2. **Environment Setup**: Configure your environment with the necessary Hugging Face token for access to datasets and models.

3. **Library Import**: Import required Python libraries for model fine-tuning and manipulation.

4. **Fine-Tuning Techniques**:
   - **Supervised Fine-Tuning (SFT)**: Train the model on a dataset of instructions and responses.
   - **LoRA**: Introduce adapters in certain layers to train only a fraction of the model's weights, reducing computational costs.
   - **QLoRA**: Employ quantization in conjunction with LoRA, further reducing memory requirements by using 4-bit precision for the model's weights.

5. **Dataset Preparation**: Load the music dataset from Hugging Face and preprocess it for training.

6. **Model Fine-Tuning**:
   - Load the base Llama 2 model.
   - Apply the chosen fine-tuning technique (LoRA/QLoRA).
   - Train the model using the prepared dataset.

7. **Evaluation and Application**: Test the fine-tuned model's performance on music-related tasks and integrate it into your application.

8. **Model and Tokenizer Pushing**: Upload the fine-tuned model and tokenizer to the Hugging Face Hub for easy access and sharing.

This guide provides a step-by-step approach to fine-tuning Llama 2 using advanced techniques like LoRA and QLoRA, tailored for tasks in the music domain.





