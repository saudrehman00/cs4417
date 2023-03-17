import torch
from transformers import AutoTokenizer, GPT2LMHeadModel
tokenizer = AutoTokenizer.from_pretrained("gpt2")
# add the EOS token as PAD token to avoid warnings
model = GPT2LMHeadModel.from_pretrained(
    "gpt2", pad_token_id=tokenizer.eos_token_id)
torch.manual_seed(50)

# encode context the generation is conditioned on
input_ids = tokenizer.encode(
    'In a land far away', return_tensors='pt')

# generate text until the output length (which includes the context length) reaches 250
greedy_output = model.generate(input_ids, max_length=500, num_beams=5, no_repeat_ngram_size=3, num_return_sequences=1)

print("My GPT-2 Story:")
print("---------------")
print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))
