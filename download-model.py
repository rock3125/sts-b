from sentence_transformers import SentenceTransformer

# model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
# modal.save('./model-xlm/')

model = SentenceTransformer('stsb-roberta-large')
model.save('./model-roberta/')
