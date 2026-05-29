# This tokenizer accepts a lists of words (vocab) during initialization. Each word is assigned an integer. 
# Encoding splits the text in words and maps each word to int based on vocab
# This also stores reverse map from int to word which converts each int back to word used for decoding. Its that simple
import re
class SimpleTokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { val:key for key,val in vocab.items() }

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\'“”‘’]|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        preprocessed = [item if item in self.str_to_int
                        else "<|unk|>" for item in preprocessed]
        
        ids = [ self.str_to_int[word] for word in preprocessed]
        return ids
        
    def decode(self,ids):
        text = " ".join([ self.int_to_str[id] for id in ids])
        text = re.sub(r'\s+([,.:;?_!"()\'“”‘’])', r'\1', text)
        return text

