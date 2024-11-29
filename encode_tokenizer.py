from khmernltk import word_tokenize
from sentence_transformers import SentenceTransformer

# Define Khmer sentences
sentences = ["សួស្ដី"]

# Tokenize each sentence using khmernltk
tokenized_sentences = [" ".join(word_tokenize(sentence)) for sentence in sentences]

# Load the SentenceTransformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Encode the tokenized sentences
embeddings = model.encode(tokenized_sentences)

# Print the embeddings
print(embeddings)
