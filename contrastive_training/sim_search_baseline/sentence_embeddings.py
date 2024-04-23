from transformers import AutoTokenizer, AutoModel
import torch


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_sentence_embeddings(sentences, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        model_output = model(**encoded_input)

    #TODO: Check if mean_pooling is recommended for xlm-r models or max_pooling, etc
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return sentence_embeddings

if __name__ == '__main__':
    sentences = ['This is an example sentence', 'Each sentence is converted']
    model_name = 'xlm-roberta-base' #Also 'xlm-roberta-large', 'facebook/xlm-roberta-xl', 'facebook/xlm-roberta-xxl' #First two models in one org and other two in other org
    embeddings = get_sentence_embeddings(sentences, model_name)
    print(embeddings)
