from sentence_transformers import SentenceTransformer, util
import numpy as np

model = SentenceTransformer('./model-xlm', device='cpu')

sentence1 = "I like Python because I can build AI applications"
sentence2 = "I like Python because I can build Artificial Intelligence applications"

# encode sentences to get their embeddings
embedding1 = model.encode(sentence1, convert_to_tensor=True)
embedding2 = model.encode(sentence2, convert_to_tensor=True)
print(embedding1.shape)

# compute similarity scores of two embeddings
cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)
print("Sentence 1:", sentence1)
print("Sentence 2:", sentence2)
print("Similarity score:", cosine_scores.item())
