from __future__ import annotations

import torch
import json
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


Pair = Tuple[str, str]
WordSymbols = Tuple[str, ...]


class BPE:
    def __init__(
        self,
        vocab_size: int = 1000,
        unk_token: str = "<UNK>",
        eow_token: str = "</w>",
        lowercase: bool = False,
    ) -> None:
        self.vocab_size = vocab_size
        self.unk_token = unk_token
        self.eow_token = eow_token
        self.lowercase = lowercase

        self.merges: List[Pair] = []
        self.merge_ranks: Dict[Pair, int] = {}
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

    def train(self, corpus: str | List[str]) -> None:
        self.merges = []
        self.merge_ranks = {}

        words = self._split_words(corpus)
        if self.lowercase:
            words = [w.lower() for w in words]

        word_freq = Counter(words)

        # Each word starts as chars + end-of-word marker
        vocab_words: Dict[WordSymbols, int] = {
            tuple(list(word) + [self.eow_token]): freq
            for word, freq in word_freq.items()
        }

        vocab = {self.unk_token, self.eow_token}
        for symbols in vocab_words:
            vocab.update(symbols)

        while len(vocab) < self.vocab_size:
            pair_stats = self._get_pair_stats(vocab_words)
            if not pair_stats:
                break

            # Deterministic tie-break: freq, then lexicographic pair
            best_pair = max(pair_stats.items(), key=lambda x: (x[1], x[0]))[0]

            self.merges.append(best_pair)
            self.merge_ranks[best_pair] = len(self.merges) - 1

            vocab_words = self._merge_pair_in_vocab(best_pair, vocab_words)
            vocab.add(best_pair[0] + best_pair[1])

        # Final vocab from resulting word symbols + unk
        final_vocab = {self.unk_token}
        for symbols in vocab_words:
            final_vocab.update(symbols)

        # Keep merged tokens as well (safe for deterministic id mapping)
        for a, b in self.merges:
            final_vocab.add(a + b)

        sorted_vocab = sorted(final_vocab)
        self.token_to_id = {tok: i for i, tok in enumerate(sorted_vocab)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}

    fit = train  # alias

    def encode_word(self, word: str) -> List[int]:
        if self.lowercase:
            word = word.lower()

        symbols = list(word) + [self.eow_token]

        while True:
            pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
            candidate_pairs = [p for p in pairs if p in self.merge_ranks]
            if not candidate_pairs:
                break

            # Apply earliest learned merge first
            best_pair = min(candidate_pairs, key=lambda p: self.merge_ranks[p])
            symbols = self._merge_pair_in_symbols(symbols, best_pair)

        unk_id = self.token_to_id[self.unk_token]
        return [self.token_to_id.get(sym, unk_id) for sym in symbols]

    def encode(self, text: str) -> List[int]:
        ids: List[int] = []
        for word in self._split_words(text):
            ids.extend(self.encode_word(word))
        return ids

    def decode(self, ids: List[int]) -> str:
        tokens = [self.id_to_token.get(i, self.unk_token) for i in ids]
        text = "".join(tokens)
        text = text.replace(self.eow_token, " ")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def save(self, path: str) -> None:
        data = {
            "vocab_size": self.vocab_size,
            "unk_token": self.unk_token,
            "eow_token": self.eow_token,
            "lowercase": self.lowercase,
            "merges": [[a, b] for a, b in self.merges],
            "token_to_id": self.token_to_id,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "BPE":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tok = cls(
            vocab_size=data["vocab_size"],
            unk_token=data["unk_token"],
            eow_token=data["eow_token"],
            lowercase=data.get("lowercase", False),
        )
        tok.merges = [tuple(p) for p in data["merges"]]
        tok.merge_ranks = {pair: i for i, pair in enumerate(tok.merges)}
        tok.token_to_id = {k: int(v) for k, v in data["token_to_id"].items()}
        tok.id_to_token = {i: t for t, i in tok.token_to_id.items()}
        return tok

    @staticmethod
    def _split_words(corpus: str | List[str]) -> List[str]:
        if isinstance(corpus, str):
            return re.findall(r"\S+", corpus)
        words: List[str] = []
        for line in corpus:
            words.extend(re.findall(r"\S+", line))
        return words

    @staticmethod
    def _get_pair_stats(vocab_words: Dict[WordSymbols, int]) -> Dict[Pair, int]:
        stats: Dict[Pair, int] = defaultdict(int)
        for symbols, freq in vocab_words.items():
            for i in range(len(symbols) - 1):
                stats[(symbols[i], symbols[i + 1])] += freq
        return dict(stats)

    @staticmethod
    def _merge_pair_in_symbols(symbols: List[str], pair: Pair) -> List[str]:
        a, b = pair
        merged: List[str] = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                merged.append(a + b)
                i += 2
            else:
                merged.append(symbols[i])
                i += 1
        return merged

    def _merge_pair_in_vocab(
        self,
        pair: Pair,
        vocab_words: Dict[WordSymbols, int],
    ) -> Dict[WordSymbols, int]:
        out: Dict[WordSymbols, int] = {}
        for symbols, freq in vocab_words.items():
            new_symbols = tuple(self._merge_pair_in_symbols(list(symbols), pair))
            out[new_symbols] = out.get(new_symbols, 0) + freq
        return out

  


DATA_PATH = "data/synthetic_data.txt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_text(path: str) -> str:  

       with open(path, "r", encoding="utf-8") as f:  
            return f.read()  


def prepare_data(path, vocab_size: int =1000, max_encoded_tokens: int = 10_000, train_ratio: float = 0.9): 
   
   text = load_text(path)  

   tokenizer = BPE(vocab_size=vocab_size) 
   tokenizer.train(text)  

   ids = tokenizer.encode(text) 
   ids = ids[:max_encoded_tokens]  

   if len(ids) < 2:  
        raise ValueError("Not enough tokens after encoding")  
   
   split_idx = int(len(ids) * train_ratio) 
   train_ids = torch.tensor(ids[:split_idx], dtype=torch.long)  
   val_ids = torch.tensor(ids[split_idx:], dtype=torch.long)  

   return tokenizer, train_ids, val_ids  



def get_batch(
    split,
    train_ids: torch.Tensor,
    val_ids: torch.Tensor,
    B: int,
    T: int,
    device: str | torch.device = DEVICE,
):

   data =  train_ids if split == "train" else val_ids 
   if len(data) <= T:  
        raise ValueError(f"{split} split too small for T={T}") 

   ix = torch.randint(0, len(data)- T, (B, ) )
   x =  torch.stack([data[i: i + T] for i in ix])  
   y = torch.stack([data[i+1:i+T+1] for i in ix])   

   return x.to(device), y.to(device)


if __name__ == "__main__":
    tokenizer, train_ids, val_ids = prepare_data(
        DATA_PATH,
        vocab_size=1200,
        max_encoded_tokens=10_000,
    )

    B, T = 32, 64

    xb, yb = get_batch("train", train_ids, val_ids, B, T)

    print("train tokens:", len(train_ids), "val tokens:", len(val_ids))
    print("xb shape:", xb.shape)
    print("yb shape:", yb.shape)
    
