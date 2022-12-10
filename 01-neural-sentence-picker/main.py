# 01-neural-sentence-picker

from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def loss(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs, labels=inputs["input_ids"])
    return outputs.loss.cpu().detach().numpy()

def get_most_likely_text(texts):
    losses = [loss(text) for text in texts]
    return(texts[losses.index(min(losses))])


texts = ['Somebody call policeman.',
         'Somebody call a policeman.']
print(get_most_likely_text(texts))

texts = ['I went home.',
         'I went to home.']
print(get_most_likely_text(texts))

texts = ["It's a beautiful town.",
         "Its a beautiful town."]
print(get_most_likely_text(texts))

texts = ["I have lived here for three weeks.",
         "I have lived here since three weeks."]
print(get_most_likely_text(texts))

texts = ["Can you tell me a little bit more about yourself.",
         "Can you tell me little bit more about yourself."]
print(get_most_likely_text(texts))
