from transformers import AutoConfig, AutoModel, AutoTokenizer

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(tokenizer)
print(type(tokenizer))

config = AutoConfig.from_pretrained(model_name)

print(config)
print(type(config))

#model = AutoModel.from_pretrained(model_name, config=config)

#print(model)
#print(type())
