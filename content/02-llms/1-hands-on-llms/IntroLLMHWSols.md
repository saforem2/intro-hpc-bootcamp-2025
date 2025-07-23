# Intro to HPC Bootcamp 2025
Sam Foreman
2025-07-22

<link rel="preconnect" href="https://fonts.googleapis.com">

``` python
'''
Uncomment below section if running on sophia jupyter notebook
'''
import os
os.environ["HTTP_PROXY"]="proxy.alcf.anl.gov:3128"
os.environ["HTTPS_PROXY"]="proxy.alcf.anl.gov:3128"
os.environ["http_proxy"]="proxy.alcf.anl.gov:3128"
os.environ["https_proxy"]="proxy.alcf.anl.gov:3128"
os.environ["ftp_proxy"]="proxy.alcf.anl.gov:3128"
```

``` python
!pip install transformers
!pip install pandas
!pip install torch
```

    Defaulting to user installation because normal site-packages is not writeable
    Requirement already satisfied: transformers in /home/avasan/.local/lib/python3.12/site-packages (4.46.0)
    Requirement already satisfied: filelock in /soft/applications/miniconda3/conda_pytorch/lib/python3.12/site-packages (from transformers) (3.15.4)
    Requirement already satisfied: huggingface-hub<1.0,>=0.23.2 in /home/avasan/.local/lib/python3.12/site-packages (from transformers) (0.26.2)
    Requirement already satisfied: numpy>=1.17 in /soft/applications/miniconda3/conda_pytorch/lib/python3.12/site-packages (from transformers) (1.26.4)
    Requirement already satisfied: packaging>=20.0 in /soft/applications/miniconda3/conda_pytorch/lib/python3.12/site-packages (from transformers) (23.2)
    Requirement already satisfied: pyyaml>=5.1 in /soft/applications/miniconda3/conda_pytorch/lib/python3.12/site-packages (from transformers) (6.0.1)
    Requirement already satisfied: regex!=2019.12.17 in /home/avasan/.local/lib/python3.12/site-packages (from transformers) (2024.9.11)
    Requirement already satisfied: requests in /soft/applications/miniconda3/conda_pytorch/lib/python3.12/site-packages (from transformers) (2.31.0)
    Requirement already satisfied: safetensors>=0.4.1 in /home/avasan/.local/lib/python3.12/site-packages (from transformers) (0.4.5)
    Requirement already satisfied: tokenizers<0.21,>=0.20 in /home/avasan/.local/lib/python3.12/site-packages (from transformers) (0.20.1)
    Requirement already satisfied: tqdm>=4.27 in /soft/applications/miniconda3/conda_pytorch/lib/python3.12/site-packages (from transformers) (4.66.2)
    Requirement already satisfied: fsspec>=2023.5.0 in /home/avasan/.local/lib/python3.12/site-packages (from huggingface-hub<1.0,>=0.23.2->transformers) (2024.10.0)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in /soft/applications/miniconda3/conda_pytorch/lib/python3.12/site-packages (from huggingface-hub<1.0,>=0.23.2->transformers) (4.12.2)
    Requirement already satisfied: charset-normalizer<4,>=2 in /soft/applications/miniconda3/conda_pytorch/lib/python3.12/site-packages (from requests->transformers) (2.0.4)
    Requirement already satisfied: idna<4,>=2.5 in /soft/applications/miniconda3/conda_pytorch/lib/python3.12/site-packages (from requests->transformers) (3.7)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /soft/applications/miniconda3/conda_pytorch/lib/python3.12/site-packages (from requests->transformers) (2.1.0)
    Requirement already satisfied: certifi>=2017.4.17 in /soft/applications/miniconda3/conda_pytorch/lib/python3.12/site-packages (from requests->transformers) (2024.7.4)
    Defaulting to user installation because normal site-packages is not writeable
    Requirement already satisfied: pandas in /home/avasan/.local/lib/python3.12/site-packages (2.2.3)
    Requirement already satisfied: numpy>=1.26.0 in /soft/applications/miniconda3/conda_pytorch/lib/python3.12/site-packages (from pandas) (1.26.4)
    Requirement already satisfied: python-dateutil>=2.8.2 in /home/avasan/.local/lib/python3.12/site-packages (from pandas) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /soft/applications/miniconda3/conda_pytorch/lib/python3.12/site-packages (from pandas) (2024.1)
    Requirement already satisfied: tzdata>=2022.7 in /home/avasan/.local/lib/python3.12/site-packages (from pandas) (2024.2)
    Requirement already satisfied: six>=1.5 in /soft/applications/miniconda3/conda_pytorch/lib/python3.12/site-packages (from python-dateutil>=2.8.2->pandas) (1.16.0)
    Defaulting to user installation because normal site-packages is not writeable
    Requirement already satisfied: torch in /soft/applications/miniconda3/conda_pytorch/lib/python3.12/site-packages (2.3.1)
    Requirement already satisfied: filelock in /soft/applications/miniconda3/conda_pytorch/lib/python3.12/site-packages (from torch) (3.15.4)
    Requirement already satisfied: typing-extensions>=4.8.0 in /soft/applications/miniconda3/conda_pytorch/lib/python3.12/site-packages (from torch) (4.12.2)
    Requirement already satisfied: sympy in /soft/applications/miniconda3/conda_pytorch/lib/python3.12/site-packages (from torch) (1.12.1)
    Requirement already satisfied: networkx in /soft/applications/miniconda3/conda_pytorch/lib/python3.12/site-packages (from torch) (3.3)
    Requirement already satisfied: jinja2 in /soft/applications/miniconda3/conda_pytorch/lib/python3.12/site-packages (from torch) (3.1.4)
    Requirement already satisfied: fsspec in /home/avasan/.local/lib/python3.12/site-packages (from torch) (2024.10.0)
    Requirement already satisfied: MarkupSafe>=2.0 in /soft/applications/miniconda3/conda_pytorch/lib/python3.12/site-packages (from jinja2->torch) (2.1.5)
    Requirement already satisfied: mpmath<1.4.0,>=1.1.0 in /soft/applications/miniconda3/conda_pytorch/lib/python3.12/site-packages (from sympy->torch) (1.3.0)

1.  Load in a generative model using the HuggingFace pipeline and
    generate text using a batch of prompts.

- Play with generative parameters such as temperature, max_new_tokens,
  and the model itself and explain the effect on the legibility of the
  model response. Try at least 4 different parameter/model combinations.

GPT2 model

``` python
from transformers import AutoTokenizer,AutoModelForCausalLM, AutoConfig
prompts = ['No, I am your',
           'Im sorry Dave, Im afraid I',
           'What is your favorite color?',
          'Forget it Jake, its',
          'You cant handle the']

from transformers import pipeline
generator = pipeline("text-generation", model="gpt2")
pipe = pipeline("text-generation", model="gpt2")
generator(prompts, max_length=20, num_return_sequences=5)
```

    Hardware accelerator e.g. GPU is available in the environment, but no `device` argument is passed to the `Pipeline` object. Model will be on CPU.
    Hardware accelerator e.g. GPU is available in the environment, but no `device` argument is passed to the `Pipeline` object. Model will be on CPU.
    Truncation was not explicitly activated but `max_length` is provided a specific value, please use `truncation=True` to explicitly truncate examples to max length. Defaulting to 'longest_first' truncation strategy. If you encode pairs of sequences (GLUE-style) with the tokenizer you can select this strategy more precisely by providing a specific strategy to `truncation`.
    Setting `pad_token_id` to `eos_token_id`:None for open-end generation.
    Setting `pad_token_id` to `eos_token_id`:None for open-end generation.
    Setting `pad_token_id` to `eos_token_id`:None for open-end generation.
    Setting `pad_token_id` to `eos_token_id`:None for open-end generation.
    Setting `pad_token_id` to `eos_token_id`:None for open-end generation.

    [[{'generated_text': "No, I am your father. I am your mother. You aren't my brother, I am"},
      {'generated_text': 'No, I am your son, your father."\n\nThere\'s something in her voice that sounds'},
      {'generated_text': 'No, I am your man."\n\nThe two boys ran off in the other direction. It'},
      {'generated_text': 'No, I am your mother."\n\n"Yay."\n\n"Yes, but..."'},
      {'generated_text': 'No, I am your opponent," she said sarcastically. "But you deserve it all."\n'}],
     [{'generated_text': 'Im sorry Dave, Im afraid I have too much to say. My friend at work came by after'},
      {'generated_text': "Im sorry Dave, Im afraid I'm not going to do it.\n\nOh man, I"},
      {'generated_text': 'Im sorry Dave, Im afraid I am going to burn yourself again. What do you want from me'},
      {'generated_text': "Im sorry Dave, Im afraid I'll go on the air, but my mom would think I'm"},
      {'generated_text': 'Im sorry Dave, Im afraid I forgot to tell you..."\n\n"It\'s only after getting'}],
     [{'generated_text': "What is your favorite color? Let us know!\n\nYou've written about that in the past"},
      {'generated_text': 'What is your favorite color? If the answer is green, what color are you choosing? Which one'},
      {'generated_text': 'What is your favorite color?\n\nWe have a few. You may know of some, but'},
      {'generated_text': 'What is your favorite color? Any time at all! ♥\n\nThis is my favorite color'},
      {'generated_text': 'What is your favorite color? Which is the best? Which can you get at a supermarket? Can'}],
     [{'generated_text': 'Forget it Jake, its the power from that damned tree.\n\nJake. Good luck.'},
      {'generated_text': "Forget it Jake, its not going to happen. I wish he'd have his own thing."},
      {'generated_text': 'Forget it Jake, its what we do now.\n\nIf you enjoyed this article, consider'},
      {'generated_text': "Forget it Jake, its just you're trying to get a good job out of this company."},
      {'generated_text': "Forget it Jake, its inescapable — you have to understand your team's offense.\n"}],
     [{'generated_text': "You cant handle the game, I'd rather be done, so it will be in the hands of"},
      {'generated_text': 'You cant handle the situation that everyone in our business, regardless of what we can do, is going'},
      {'generated_text': 'You cant handle the same problem that a small party can, a few members of a larger group would'},
      {'generated_text': 'You cant handle the pressure of a job. Because I have a job and no other work, I'},
      {'generated_text': 'You cant handle the fact that we\'ve been doing the same thing forever."\n\nThe idea that'}]]

``` python
generator(prompts, max_length=10, num_return_sequences=4)
```

    Setting `pad_token_id` to `eos_token_id`:None for open-end generation.
    Setting `pad_token_id` to `eos_token_id`:None for open-end generation.
    Setting `pad_token_id` to `eos_token_id`:None for open-end generation.
    Setting `pad_token_id` to `eos_token_id`:None for open-end generation.
    Setting `pad_token_id` to `eos_token_id`:None for open-end generation.

    [[{'generated_text': 'No, I am your god," his voice cracked'},
      {'generated_text': 'No, I am your Lord and I will go'},
      {'generated_text': 'No, I am your father, and I owe'},
      {'generated_text': 'No, I am your enemy. And you have'}],
     [{'generated_text': 'Im sorry Dave, Im afraid I might as well'},
      {'generated_text': 'Im sorry Dave, Im afraid I will be so'},
      {'generated_text': 'Im sorry Dave, Im afraid I am getting my'},
      {'generated_text': "Im sorry Dave, Im afraid I won't be"}],
     [{'generated_text': 'What is your favorite color? Let us know with'},
      {'generated_text': 'What is your favorite color? Tell us in the'},
      {'generated_text': 'What is your favorite color? We love the orange'},
      {'generated_text': 'What is your favorite color?\n\nI got'}],
     [{'generated_text': 'Forget it Jake, its like it was your'},
      {'generated_text': 'Forget it Jake, its a joke. You'},
      {'generated_text': 'Forget it Jake, its time for the rest'},
      {'generated_text': 'Forget it Jake, its not a lot of'}],
     [{'generated_text': 'You cant handle the time, will we have to'},
      {'generated_text': 'You cant handle the whole thing myself (though it'},
      {'generated_text': 'You cant handle the situation right now."\n\n'},
      {'generated_text': "You cant handle the fact that you're doing this"}]]

``` python
generator(prompts, max_length=10, num_return_sequences=4, temperature=1)
```

    Setting `pad_token_id` to `eos_token_id`:None for open-end generation.
    Setting `pad_token_id` to `eos_token_id`:None for open-end generation.
    Setting `pad_token_id` to `eos_token_id`:None for open-end generation.
    Setting `pad_token_id` to `eos_token_id`:None for open-end generation.
    Setting `pad_token_id` to `eos_token_id`:None for open-end generation.

    [[{'generated_text': 'No, I am your God. I am your'},
      {'generated_text': 'No, I am your father," she says as'},
      {'generated_text': 'No, I am your uncle or yours," answered'},
      {'generated_text': 'No, I am your best friend, I will'}],
     [{'generated_text': 'Im sorry Dave, Im afraid I need this place'},
      {'generated_text': "Im sorry Dave, Im afraid I'm not happy"},
      {'generated_text': 'Im sorry Dave, Im afraid I have a problem'},
      {'generated_text': 'Im sorry Dave, Im afraid I may never catch'}],
     [{'generated_text': 'What is your favorite color? Tell us in the'},
      {'generated_text': 'What is your favorite color? What can I do'},
      {'generated_text': 'What is your favorite color? Check out this list'},
      {'generated_text': 'What is your favorite color? Let us know!'}],
     [{'generated_text': "Forget it Jake, its not that you're"},
      {'generated_text': 'Forget it Jake, its time to quit trying'},
      {'generated_text': "Forget it Jake, its just about what's"},
      {'generated_text': 'Forget it Jake, its in those days what'}],
     [{'generated_text': 'You cant handle the pressure of what I would do'},
      {'generated_text': 'You cant handle the thought of you taking out my'},
      {'generated_text': 'You cant handle the same situation. As you can'},
      {'generated_text': "You cant handle the game of 'Tin-"}]]

Llama 3.2 1B

``` python
from huggingface_hub import login
hf_token = "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"
login(token=hf_token, add_to_git_credential=True)
from transformers import AutoTokenizer,AutoModelForCausalLM, AutoConfig
from transformers import pipeline

pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-1B", device=0)
```

``` python
messages = [
    "Star Wars: No, I am your",
    "Space Odyssey: Im sorry Dave, Im afraid I",
    "Monty Python: What is your favorite ",
    "Chinatown: Forget it Jake, its",
    "A few good men: You cant handle the",
]
pipe(messages, max_length=20, num_return_sequences=5, temperature=1)
```

    Setting `pad_token_id` to `eos_token_id`:None for open-end generation.
    Setting `pad_token_id` to `eos_token_id`:None for open-end generation.
    Setting `pad_token_id` to `eos_token_id`:None for open-end generation.
    Setting `pad_token_id` to `eos_token_id`:None for open-end generation.
    Setting `pad_token_id` to `eos_token_id`:None for open-end generation.

    [[{'generated_text': 'Star Wars: No, I am your father, Luke\nA few days ago I found myself'},
      {'generated_text': 'Star Wars: No, I am your father\nA very sweet and touching, in a way'},
      {'generated_text': 'Star Wars: No, I am your father. I am your father.\nStar Wars: I'},
      {'generated_text': 'Star Wars: No, I am your father is a 2014 Australian sci-fi action comedy'},
      {'generated_text': "Star Wars: No, I am your father... and no, I'm not a cop ("}],
     [{'generated_text': 'Space Odyssey: Im sorry Dave, Im afraid I have to disagree.\nI don’t think that'},
      {'generated_text': 'Space Odyssey: Im sorry Dave, Im afraid I have to do this.\nSpace Odyssey: Im'},
      {'generated_text': "Space Odyssey: Im sorry Dave, Im afraid I'm not going to be a part of this"},
      {'generated_text': 'Space Odyssey: Im sorry Dave, Im afraid I dont buy your reasoning. I love space exploration'},
      {'generated_text': "Space Odyssey: Im sorry Dave, Im afraid I cant help you. It's a matter of"}],
     [{'generated_text': 'Monty Python: What is your favorite 20 minutes in the whole world? That’s right'},
      {'generated_text': 'Monty Python: What is your favorite 2017 Christmas movie?\nChristmas. It’s a'},
      {'generated_text': 'Monty Python: What is your favorite 1 1 1 1 1 '},
      {'generated_text': 'Monty Python: What is your favorite 2.5 hour long stand up?\nDiscussion in'},
      {'generated_text': 'Monty Python: What is your favorite 2000s era joke?\nI think I would'}],
     [{'generated_text': 'Chinatown: Forget it Jake, its the real deal\nIt seems as if Chinatown'},
      {'generated_text': 'Chinatown: Forget it Jake, its history!\nThis is not what Chinatown looks like'},
      {'generated_text': 'Chinatown: Forget it Jake, its the East Village\nOn Friday night, the East'},
      {'generated_text': 'Chinatown: Forget it Jake, its on the way down\nChinatown - Photo'},
      {'generated_text': 'Chinatown: Forget it Jake, its Chinatown, we will not be bullied by this'}],
     [{'generated_text': 'A few good men: You cant handle the truth\nI saw an advertisement for an interview with'},
      {'generated_text': 'A few good men: You cant handle the truth, it turns out. Oh, and don'},
      {'generated_text': 'A few good men: You cant handle the truth\nI\'ve noticed that the term "you'},
      {'generated_text': 'A few good men: You cant handle the truth\nOn Monday, I wrote the blog "'},
      {'generated_text': 'A few good men: You cant handle the truth.\nA few good men: You cant handle'}]]

``` python
pipe(messages, max_length=20, num_return_sequences=5, temperature=0.05)
```

    Setting `pad_token_id` to `eos_token_id`:None for open-end generation.
    Setting `pad_token_id` to `eos_token_id`:None for open-end generation.
    Setting `pad_token_id` to `eos_token_id`:None for open-end generation.
    Setting `pad_token_id` to `eos_token_id`:None for open-end generation.
    Setting `pad_token_id` to `eos_token_id`:None for open-end generation.

    [[{'generated_text': 'Star Wars: No, I am your father\nThe Star Wars franchise has been around for over'},
      {'generated_text': 'Star Wars: No, I am your father\nThe Star Wars franchise has been around for over'},
      {'generated_text': 'Star Wars: No, I am your father\nThe Star Wars franchise has been around for over'},
      {'generated_text': 'Star Wars: No, I am your father\nThe Star Wars franchise has been around for over'},
      {'generated_text': 'Star Wars: No, I am your father\nI am your father. I am your father'}],
     [{'generated_text': "Space Odyssey: Im sorry Dave, Im afraid I can't do that.\nSpace Odyssey: Im"},
      {'generated_text': "Space Odyssey: Im sorry Dave, Im afraid I can't do that.\nSpace Odyssey: Im"},
      {'generated_text': "Space Odyssey: Im sorry Dave, Im afraid I can't do that.\nSpace Odyssey: Im"},
      {'generated_text': "Space Odyssey: Im sorry Dave, Im afraid I can't do that.\nSpace Odyssey: Im"},
      {'generated_text': "Space Odyssey: Im sorry Dave, Im afraid I can't do that.\nSpace Odyssey: Im"}],
     [{'generated_text': 'Monty Python: What is your favorite 70s movie?\nWhat is your favorite 70'},
      {'generated_text': 'Monty Python: What is your favorite 70s movie?\nWhat is your favorite 70'},
      {'generated_text': "Monty Python: What is your favorite 70s movie?\nI'm a big fan of"},
      {'generated_text': 'Monty Python: What is your favorite 70s movie?\nWhat is your favorite 70'},
      {'generated_text': "Monty Python: What is your favorite 70s movie?\nI'm a big fan of"}],
     [{'generated_text': 'Chinatown: Forget it Jake, its the 21st century\nThe 21st'},
      {'generated_text': 'Chinatown: Forget it Jake, its the 21st century\nThe 21st'},
      {'generated_text': 'Chinatown: Forget it Jake, its the 21st century\nThe 21st'},
      {'generated_text': 'Chinatown: Forget it Jake, its the 21st century\nThe 21st'},
      {'generated_text': 'Chinatown: Forget it Jake, its the 21st century\nThe 21st'}],
     [{'generated_text': 'A few good men: You cant handle the truth\nA few good men: You cant handle'},
      {'generated_text': 'A few good men: You cant handle the truth\nA few good men: You cant handle'},
      {'generated_text': 'A few good men: You cant handle the truth\nA few good men: You cant handle'},
      {'generated_text': 'A few good men: You cant handle the truth\nA few good men: You cant handle'},
      {'generated_text': 'A few good men: You cant handle the truth\nA few good men: You cant handle'}]]

2.  Load in 2 models of different parameter size (e.g. GPT2,
    meta-llama/Llama-2-7b-chat-hf, or distilbert/distilgpt2) and analyze
    the BertViz for each. How does the attention mechanisms change
    depending on model size?

GPT2

``` python
!pip install bertviz
```

    Defaulting to user installation because normal site-packages is not writeable
    Collecting bertviz
      Using cached bertviz-1.4.0-py3-none-any.whl.metadata (19 kB)
    Requirement already satisfied: transformers>=2.0 in /home/avasan/.local/lib/python3.12/site-packages (from bertviz) (4.46.0)
    Requirement already satisfied: torch>=1.0 in /soft/applications/miniconda3/conda_pytorch/lib/python3.12/site-packages (from bertviz) (2.3.1)
    Requirement already satisfied: tqdm in /soft/applications/miniconda3/conda_pytorch/lib/python3.12/site-packages (from bertviz) (4.66.2)
    Collecting boto3 (from bertviz)
      Downloading boto3-1.35.62-py3-none-any.whl.metadata (6.7 kB)
    Requirement already satisfied: requests in /soft/applications/miniconda3/conda_pytorch/lib/python3.12/site-packages (from bertviz) (2.31.0)
    Requirement already satisfied: regex in /home/avasan/.local/lib/python3.12/site-packages (from bertviz) (2024.9.11)
    Collecting sentencepiece (from bertviz)
      Downloading sentencepiece-0.2.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (7.7 kB)
    Requirement already satisfied: filelock in /soft/applications/miniconda3/conda_pytorch/lib/python3.12/site-packages (from torch>=1.0->bertviz) (3.15.4)
    Requirement already satisfied: typing-extensions>=4.8.0 in /soft/applications/miniconda3/conda_pytorch/lib/python3.12/site-packages (from torch>=1.0->bertviz) (4.12.2)
    Requirement already satisfied: sympy in /soft/applications/miniconda3/conda_pytorch/lib/python3.12/site-packages (from torch>=1.0->bertviz) (1.12.1)
    Requirement already satisfied: networkx in /soft/applications/miniconda3/conda_pytorch/lib/python3.12/site-packages (from torch>=1.0->bertviz) (3.3)
    Requirement already satisfied: jinja2 in /soft/applications/miniconda3/conda_pytorch/lib/python3.12/site-packages (from torch>=1.0->bertviz) (3.1.4)
    Requirement already satisfied: fsspec in /home/avasan/.local/lib/python3.12/site-packages (from torch>=1.0->bertviz) (2024.10.0)
    Requirement already satisfied: huggingface-hub<1.0,>=0.23.2 in /home/avasan/.local/lib/python3.12/site-packages (from transformers>=2.0->bertviz) (0.26.2)
    Requirement already satisfied: numpy>=1.17 in /soft/applications/miniconda3/conda_pytorch/lib/python3.12/site-packages (from transformers>=2.0->bertviz) (1.26.4)
    Requirement already satisfied: packaging>=20.0 in /soft/applications/miniconda3/conda_pytorch/lib/python3.12/site-packages (from transformers>=2.0->bertviz) (23.2)
    Requirement already satisfied: pyyaml>=5.1 in /soft/applications/miniconda3/conda_pytorch/lib/python3.12/site-packages (from transformers>=2.0->bertviz) (6.0.1)
    Requirement already satisfied: safetensors>=0.4.1 in /home/avasan/.local/lib/python3.12/site-packages (from transformers>=2.0->bertviz) (0.4.5)
    Requirement already satisfied: tokenizers<0.21,>=0.20 in /home/avasan/.local/lib/python3.12/site-packages (from transformers>=2.0->bertviz) (0.20.1)
    Collecting botocore<1.36.0,>=1.35.62 (from boto3->bertviz)
      Downloading botocore-1.35.62-py3-none-any.whl.metadata (5.7 kB)
    Collecting jmespath<2.0.0,>=0.7.1 (from boto3->bertviz)
      Using cached jmespath-1.0.1-py3-none-any.whl.metadata (7.6 kB)
    Collecting s3transfer<0.11.0,>=0.10.0 (from boto3->bertviz)
      Downloading s3transfer-0.10.3-py3-none-any.whl.metadata (1.7 kB)
    Requirement already satisfied: charset-normalizer<4,>=2 in /soft/applications/miniconda3/conda_pytorch/lib/python3.12/site-packages (from requests->bertviz) (2.0.4)
    Requirement already satisfied: idna<4,>=2.5 in /soft/applications/miniconda3/conda_pytorch/lib/python3.12/site-packages (from requests->bertviz) (3.7)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /soft/applications/miniconda3/conda_pytorch/lib/python3.12/site-packages (from requests->bertviz) (2.1.0)
    Requirement already satisfied: certifi>=2017.4.17 in /soft/applications/miniconda3/conda_pytorch/lib/python3.12/site-packages (from requests->bertviz) (2024.7.4)
    Requirement already satisfied: python-dateutil<3.0.0,>=2.1 in /home/avasan/.local/lib/python3.12/site-packages (from botocore<1.36.0,>=1.35.62->boto3->bertviz) (2.8.2)
    Requirement already satisfied: MarkupSafe>=2.0 in /soft/applications/miniconda3/conda_pytorch/lib/python3.12/site-packages (from jinja2->torch>=1.0->bertviz) (2.1.5)
    Requirement already satisfied: mpmath<1.4.0,>=1.1.0 in /soft/applications/miniconda3/conda_pytorch/lib/python3.12/site-packages (from sympy->torch>=1.0->bertviz) (1.3.0)
    Requirement already satisfied: six>=1.5 in /soft/applications/miniconda3/conda_pytorch/lib/python3.12/site-packages (from python-dateutil<3.0.0,>=2.1->botocore<1.36.0,>=1.35.62->boto3->bertviz) (1.16.0)
    Using cached bertviz-1.4.0-py3-none-any.whl (157 kB)
    Downloading boto3-1.35.62-py3-none-any.whl (139 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 139.2/139.2 kB 14.1 MB/s eta 0:00:00
    Downloading sentencepiece-0.2.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.3 MB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.3/1.3 MB 75.0 MB/s eta 0:00:00
    Downloading botocore-1.35.62-py3-none-any.whl (12.8 MB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 12.8/12.8 MB 141.3 MB/s eta 0:00:0000:010:01
    Using cached jmespath-1.0.1-py3-none-any.whl (20 kB)
    Downloading s3transfer-0.10.3-py3-none-any.whl (82 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 82.6/82.6 kB 8.1 MB/s eta 0:00:00
    Installing collected packages: sentencepiece, jmespath, botocore, s3transfer, boto3, bertviz
    Successfully installed bertviz-1.4.0 boto3-1.35.62 botocore-1.35.62 jmespath-1.0.1 s3transfer-0.10.3 sentencepiece-0.2.0

``` python
from transformers import AutoTokenizer, AutoModel, utils, AutoModelForCausalLM

from bertviz import model_view
utils.logging.set_verbosity_error()  # Suppress standard warnings

model_name = 'openai-community/gpt2'
input_text = "Forget it Jake, its"
model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer.encode(input_text, return_tensors='pt')  # Tokenize input text
outputs = model(inputs)  # Run model
attention = outputs[-1]  # Retrieve attention from model outputs
tokens = tokenizer.convert_ids_to_tokens(inputs[0])  # Convert input ids to token strings
model_view(attention, tokens)  # Display model view
```

<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js"></script>

      
        <div id="bertviz-c742efe0c66a4505b24f9eba0b6c5020" style="font-family:'Helvetica Neue', Helvetica, Arial, sans-serif;">
            <span style="user-select:none">
                &#10;            </span>
            <div id='vis'></div>
        </div>
    &#10;

    <IPython.core.display.Javascript object>

``` python
from transformers import AutoTokenizer, AutoModel, utils, AutoModelForCausalLM

from bertviz import model_view
utils.logging.set_verbosity_error()  # Suppress standard warnings

model_name = "meta-llama/Llama-3.2-1B"
input_text = "Forget it Jake, its"
model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer.encode(input_text, return_tensors='pt')  # Tokenize input text
outputs = model(inputs)  # Run model
attention = outputs[-1]  # Retrieve attention from model outputs
tokens = tokenizer.convert_ids_to_tokens(inputs[0])  # Convert input ids to token strings
model_view(attention, tokens)  # Display model view
```

<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js"></script>

      
        <div id="bertviz-d6e48ed10ba04422bc2836743d6f1f9b" style="font-family:'Helvetica Neue', Helvetica, Arial, sans-serif;">
            <span style="user-select:none">
                &#10;            </span>
            <div id='vis'></div>
        </div>
    &#10;

    <IPython.core.display.Javascript object>

We can see that loading in the llama model drastically increases the
number of attention mechanisms. Although the attention mechanisms look
quite similar in a lot of the heads/layers (a lot of attention to
“\<\|begin_of_text\|\>”), due to the large number of heads + layers,
there is variability in a few heads.
