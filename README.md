---
license: gemma
library_name: peft
tags:
- generated_from_trainer
base_model: google/paligemma-3b-pt-224
model-index:
- name: paligemma_VQAMed
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# paligemma_VQAMed2019

This model is a fine-tuned version of [google/paligemma-3b-pt-224](https://huggingface.co/google/paligemma-3b-pt-224) on the [VQAMed 2019](https://zenodo.org/records/10499039) dataset.

## How to use

To use the model, follow the [colab notebook](https://colab.research.google.com/drive/1SfrNNHE32k9kBWdR6U0DQr4LI_AVIAb1?usp=sharing). 
Below is a quick example.

To ensure you have the latest version of Transformers, install it using the following command: 

```bash
!pip install -qU git+https://github.com/huggingface/transformers.git
```

```python
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
import torch
from PIL import Image
import requests

processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224")
model = PaliGemmaForConditionalGeneration.from_pretrained("MahmoudRox/Paligemma_VQAMED2019")

prompt = "Which part of the body is in the picture?" #your question
image_file = "https://prod-images-static.radiopaedia.org/images/9289883/1c20962e46c92ee83a3f551adb24fa_big_gallery.jpg" #your image
raw_image = Image.open(requests.get(image_file, stream=True).raw)

def generate_response(prompt, image):
  inputs = processor(images=image, text=prompt, return_tensors="pt")

  # Check if the attention mask needs to be inverted
  attention_mask = inputs['attention_mask']
  if torch.max(attention_mask) == 1:
      attention_mask = 1 - attention_mask

  # Generate a response
  outputs = model.generate(
      input_ids=inputs['input_ids'],
      attention_mask=attention_mask,
      pixel_values=inputs['pixel_values'],
      max_new_tokens=1,
      no_repeat_ngram_size=2
  )
  
  # Decode and print the response
  decoded_response = processor.decode(outputs[0], skip_special_tokens=True)[len(prompt):]
  return decoded_response

print(generate_response(prompt, raw_image))
#spine
```

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 4
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 4
- total_train_batch_size: 16
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 2
- num_epochs: 2

### Framework versions

- PEFT 0.11.1
- Transformers 4.42.0.dev0
- Pytorch 2.3.0+cu121
- Datasets 2.19.2
- Tokenizers 0.19.1
