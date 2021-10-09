from sentence_transformers import SentenceTransformer

model = SentenceTransformer('stsb-roberta-large')
model.save('./model/')
