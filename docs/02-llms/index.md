# Introduction to Large Language Models (LLMs)
Sam Foreman
2025-07-15

<link rel="preconnect" href="https://fonts.googleapis.com">

- [Contents](#contents)
- [Natural Language Processing (NLP)](#natural-language-processing-nlp)
- [Large Language Models (LLMs)](#large-language-models-llms)

## Contents

- 📂 [02-llms/](index.qmd)
  - [00-intro-to-llms](00-intro-to-llms/index.qmd)
  - [01-hands-on-llms](01-hands-on-llms/index.ipynb)
  - [02-prompt-engineering](02-prompt-engineering/index.qmd)
  - [05-advanced-llms](05-advanced-llms/index.qmd)
  - [06-parallel-training](06-parallel-training/index.qmd)
  - [07-shakespeare-example](07-shakespeare-example/index.qmd)
  - [08-shakespeare-example-colab](08-shakespeare-example-colab/index.ipynb)
  - [09-rag-tutorial](09-rag-tutorial/index.qmd)
  - [10-evaluating-llms](10-evaluating-llms/index.qmd)

## Natural Language Processing (NLP)

Large Language Models (LLMs) are a subset of Natural Language Processing
(NLP) techniques that focus on understanding and generating human
language. NLP is a field of linguistics / artificial intelligence that
enables computers to interpret, understand, and respond to human
language in a way that is both meaningful and useful.

The following is a list of common NLP tasks, with some examples:

- **Classifying whole sentences**: Getting the sentiment of a review,
  detecting if an email is spam, determining if a sentence is
  gramatically correct or whether two sentences are logically related or
  not.
- **Classifying each word in a sentence**: Identifying the grammatical
  components of a sentence (noun, verb, adjectvie, …), or the named
  entities (person, location, organization, …).
- **Generating Text**: Completing a prompt with auto-generated text,
  filling in the blanks in a text with masked words
- **Extracting an answer from a text**: Given a question and a context,
  extracting the answer to the question based on the information
  provided in the context.
- **Generating a new sentence from an input text**: Translating a text
  into another language, summarizing a text

## Large Language Models (LLMs)

> A large lanuage model (LLM) is an AI model trained on massive amounts
> of text data that can understand and generate human-like text,
> recognize patterns in language, and perform a wide variety of language
> tasks without task-specific training.  
> They represent a significant advancement in the field of natural
> language processing (NLP) (Face 2022).

> [!WARNING]
>
> ### 🚧 Warning
>
> While LLMs are are able to generate (what appears to be) human-like
> text, they are not sentient, and do not have an understanding of the
> world in the way that humans do. They are trained to predict the next
> word in a sentence based on the context of the words that come before
> it, and can generate text that is coherent and relevant to the input
> they receive. However, they do not have a true understanding of the
> meaning of the words they generate, and can sometimes produce text
> that is nonsensical or irrelevant to the input.

Even with the advances in LLMs, many fundamental challenges remain.
These include understanding ambiguity, cultural context, sarcasm and
humor. LLMs address these challenges through massive training on diverse
datasets, but still often fall short of human-level understanding in
many complex scenarios.

<div id="refs" class="references csl-bib-body hanging-indent"
entry-spacing="0">

<div id="ref-huggingfacecourse" class="csl-entry">

Face, Hugging. 2022. “The Hugging Face Course, 2022.”
<https://huggingface.co/course>.

</div>

</div>
