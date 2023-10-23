from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda"

model = AutoModelForCausalLM.from_pretrained("model_name", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("model_name")

while True:
    input_text = input("You: ")
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    output = model.generate(input_ids)

    output_text = tokenizer.decode(output[0], skip_special_tokens=True)

    print(f"Model: {output_text}")

    
