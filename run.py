from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda"

model = AutoModelForCausalLM.from_pretrained("Mistral-7B-v0.1", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("Mistral-7B-v0.1")

model = model.to("cuda")

while True:
    input_text = input("You: ")
    prompt = f"Q: {input_text} A:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")

    output = model.generate(input_ids, max_length=input_ids.shape[1] + 1024, pad_token_id=tokenizer.eos_token_id)

    output_text = tokenizer.decode(output[0], skip_special_tokens=True)

    print(f"Model: {output_text}")


