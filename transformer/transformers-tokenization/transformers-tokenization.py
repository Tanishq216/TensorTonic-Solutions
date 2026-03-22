import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """ A word-level tokenizer with special tokens. """
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    def build_vocab(self, texts: List[str]) -> None:
        """ Build vocabulary from a list of texts. """
        # 1. Initialize with special tokens in the required order
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        
        for i, token in enumerate(special_tokens):
            self.word_to_id[token] = i
            self.id_to_word[i] = token
            
        current_id = len(special_tokens)
        
        # 2. Extract unique words from the training texts
        for text in texts:
            # Simple whitespace splitting
            words = text.split()
            for word in words:
                if word not in self.word_to_id:
                    self.word_to_id[word] = current_id
                    self.id_to_word[current_id] = word
                    current_id += 1
        
        self.vocab_size = current_id

    def encode(self, text: str) -> List[int]:
        """ Convert text to list of token IDs. Use UNK for unknown words. """
        words = text.split()
        # Use .get() with the ID for <UNK> (which is 1) as the default
        return [self.word_to_id.get(word, self.word_to_id[self.unk_token]) for word in words]

    def decode(self, ids: List[int]) -> str:
        """ Convert list of token IDs back to text. """
        # Join words with spaces, defaulting to <UNK> if an ID is missing (though rare)
        return " ".join([self.id_to_word.get(i, self.unk_token) for i in ids])