from transformers import BertTokenizer, BertConfig, BertModel
import torch

class TextProcessor():
    def process(text):
        config = BertConfig()
        config.output_hidden_states = True
        model = BertModel(config)
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        max_length = 50
        encoded = tokenizer.batch_encode_plus([text], max_length=max_length, padding='max_length', truncation=True)
        encoded = {key:torch.LongTensor(value) for key, value in encoded.items()}
        with torch.no_grad():
            processed_text = model(**encoded).last_hidden_state.swapaxes(1,2)
        processed_text = processed_text.squeeze(0)
        return processed_text[None]