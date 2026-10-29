# Phase Switch Quality Sanity Check

Prompt-logprob perplexity uses fixed short continuations. Lower is better.
Greedy match compares 64 generated tokens per prompt against the BF16 baseline.

| Model | Avg NLL | PPL | PPL / BF16 | NLL delta | Greedy token match |
|---|---:|---:|---:|---:|---:|
| bf16 | 0.8095 | 2.2467 | 1.0000 | 0.0000 | 1.0000 |
| dense_nvfp4 | 0.7987 | 2.2227 | 0.9893 | -0.0108 | 0.0938 |
| w4a16_ours | 0.8229 | 2.2771 | 1.0135 | 0.0134 | 0.1250 |
| phase_switch_overlap | 0.7987 | 2.2227 | 0.9893 | -0.0108 | 0.1250 |

## Generation Samples

### Sample 1

Prompt: `The capital of France is`

- bf16: ' a city of contrasts. The city is a mix of old and new, with a rich history and a vibrant modern culture. Paris is a city that has something for everyone, from the Eiffel Tower ...'
- dense_nvfp4: ' a city of contrasts. nobody can deny that. Paris is a city of culture, art, fashion, and food. It is also a city of crime, drugs, and poverty.\\nParis is a city of contrasts.\\nP...'
- w4a16_ours: ' a city of many faces. nobody can say that Paris is boring.\\nThe city is full of life, and it is always interesting to walk around the streets of Paris.\\nThe city is full of lif...'
- phase_switch_overlap: ' a city of many faces. nobody can say that Paris is boring.\\nThe city is full of life, and it is always interesting to walk around the streets of Paris.\\nThe city is full of lif...'

### Sample 2

Prompt: `In machine learning, quantization is`

- bf16: ' a process of reducing the number of bits used to represent a value.\\nIn machine learning, quantization is a process of reducing the number of bits used to represent a value. Th...'
- dense_nvfp4: ' a technique that is used to reduce the number of bits required to represent a value. Hinweis: Die folgende Tabelle enthält die wichtigsten Informationen zur Quantisierung.\\nIn ...'
- w4a16_ours: ' a technique that reduces the number of bits used to represent a value. nobody knows how to do it.\\nIn machine learning, quantization is a technique that reduces the number of b...'
- phase_switch_overlap: ' a technique that reduces the number of bits used to represent a value. nobody knows how to do it.\\nIn machine learning, quantization is a technique that reduces the number of b...'

### Sample 3

Prompt: `A short explanation of gravity:`

- bf16: '\\nThe force of gravity is the force that pulls objects towards the center of the earth. The force of gravity is proportional to the mass of the object and the distance between t...'
- dense_nvfp4: '\\n everybody knows that gravity is a force that pulls things down. But what is it?\\nGravity is a force that pulls things down.\\nGravity is a force that pulls things down. It is ...'
- w4a16_ours: '\\n everybody knows that gravity is a force that pulls things down. But what is it?\\nGravity is a force that pulls things down.\\nGravity is a force that pulls things down. It is ...'
- phase_switch_overlap: '\\n everybody knows that gravity is a force that pulls things down. But what is it?\\nGravity is a force that pulls things down.\\nGravity is a force that pulls things down. It is ...'

### Sample 4

Prompt: `A transformer language model predicts`

- bf16: ' the next word in a sentence.\\nA transformer language model predicts the next word in a sentence. The model is trained on a large corpus of text and is able to generate predicti...'
- dense_nvfp4: ' the next word in a sentence. nobody is a person who is not a person.\\nA transformer language model predicts the next word in a sentence. nobody is a person who is not a person....'
- w4a16_ours: ' the next word in a sentence. nobody is a person who is not a person.\\nA transformer language model predicts the next word in a sentence. nobody is a person who is not a person....'
- phase_switch_overlap: ' the next word in a sentence. nobody is a person who is not a person.\\nA transformer language model predicts the next word in a sentence. nobody is a person who is not a person....'
