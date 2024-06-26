from transformers import DistilBertTokenizer, DistilBertModel
import torch

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

def get_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state


def get_bert_embeddings(text):
    # Tokenize the input text and convert to input IDs
    encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    # Get the embeddings from BERT
    with torch.no_grad():
        output = model(**encoded_input)
    # Retrieve the embeddings for the tokens (ignoring special tokens)
    embeddings = output.last_hidden_state[0, 1:-1, :]  # Ignore [CLS] and [SEP]
    return embeddings.numpy()

def extract_keywords(text, num_keywords=5):
    # Get embeddings from BERT
    embeddings = get_bert_embeddings(text)
    # Use KMeans to cluster embeddings
    kmeans = KMeans(n_clusters=num_keywords)
    kmeans.fit(embeddings)
    # Find the cluster centers
    closest_idxs = np.array([np.argmin(np.linalg.norm(embeddings - center, axis=1)) for center in kmeans.cluster_centers_])
    # Decode the tokens to get keywords
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(text)['input_ids'][1:-1])  # Exclude [CLS] and [SEP]
    keywords = [tokens[i] for i in closest_idxs]
    return keywords

text = "BERT models are widely used in the field of natural language processing. They are especially useful for tasks like text classification, named entity recognition, and keyword extraction."
keywords = extract_keywords(text, num_keywords=5)
print("Extracted Keywords:", keywords)




