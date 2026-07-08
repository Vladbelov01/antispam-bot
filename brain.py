import json
import os
import hashlib
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


os.makedirs("data", exist_ok=True)

SPAM_FILE = "data/learned_spam.json"
HAM_FILE = "data/learned_ham.json"
BANNED_FILE = "data/banned_users.json"
WHITELIST_FILE = "data/whitelist.json"


@dataclass
class Example:
    text: str
    text_hash: str
    timestamp: str
    confidence: float
    detectors_agreed: int
    detector_scores: Dict[str, float]
    user_id: Optional[int] = None
    chat_id: Optional[int] = None


class Brain:
    CONSENSUS_STRONG = 4
    CONSENSUS_MODERATE = 3
    
    def __init__(self):
        self.embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        self.spam_examples: List[Example] = []
        self.ham_examples: List[Example] = []
        self.banned_users: set = set()
        self.whitelist: set = set()
        
        self._load_all()
        self._rebuild_embeddings()
    
    def _load_all(self):
        for path, target in [(SPAM_FILE, 'spam'), (HAM_FILE, 'ham'), 
                             (BANNED_FILE, 'banned'), (WHITELIST_FILE, 'whitelist')]:
            if not os.path.exists(path):
                continue
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if target == 'spam':
                    self.spam_examples = [Example(**d) for d in data]
                elif target == 'ham':
                    self.ham_examples = [Example(**d) for d in data]
                elif target == 'banned':
                    self.banned_users = set(data)
                elif target == 'whitelist':
                    self.whitelist = {tuple(d) for d in data}
    
    def _save(self, data, path):
        with open(path, 'w', encoding='utf-8') as f:
            if isinstance(data, list) and data and hasattr(data[0], '__dataclass_fields__'):
                json.dump([asdict(d) for d in data], f, ensure_ascii=False, indent=2)
            else:
                json.dump(list(data), f)
    
    def _rebuild_embeddings(self):
        if self.spam_examples:
            self.spam_embeddings = self.embedder.encode([e.text for e in self.spam_examples])
        else:
            self.spam_embeddings = None
    
    def _hash(self, text: str) -> str:
        return hashlib.md5(text.lower().strip().encode()).hexdigest()
    
    def _is_duplicate(self, text: str) -> bool:
        h = self._hash(text)
        return any(e.text_hash == h for e in self.spam_examples)
    
    def _is_similar(self, text: str) -> bool:
        if not self.spam_examples:
            return False
        emb = self.embedder.encode([text])
        sims = cosine_similarity(emb, self.spam_embeddings)[0]
        return float(np.max(sims)) > 0.90
    
    def add_spam(self, text: str, confidence: float, agreed: int,
                 scores: Dict[str, float], user_id=None, chat_id=None) -> bool:
        if not text or len(text.strip()) < 5:
            return False
        if self._is_duplicate(text) or self._is_similar(text):
            return False
        
        ex = Example(
            text=text.strip(),
            text_hash=self._hash(text),
            timestamp=datetime.now().isoformat(),
            confidence=confidence,
            detectors_agreed=agreed,
            detector_scores=scores,
            user_id=user_id,
            chat_id=chat_id
        )
        self.spam_examples.append(ex)
        self._save(self.spam_examples, SPAM_FILE)
        self._rebuild_embeddings()
        return True
    
    def add_ham(self, text: str, user_id=None, chat_id=None):
        if not text or len(text.strip()) < 5:
            return
        ex = Example(
            text=text.strip(),
            text_hash=self._hash(text),
            timestamp=datetime.now().isoformat(),
            confidence=1.0,
            detectors_agreed=0,
            detector_scores={},
            user_id=user_id,
            chat_id=chat_id
        )
        self.ham_examples.append(ex)
        self._save(self.ham_examples, HAM_FILE)
    
    def consensus(self, results: Dict[str, float]):
        agreed = sum(1 for s in results.values() if s >= 0.6)
        total = len(results)
        avg = sum(results.values()) / total if total else 0
        weighted = sum(s for s in results.values() if s >= 0.6)
        weighted = weighted / agreed if agreed else 0
        confidence = avg * 0.3 + weighted * 0.5 + (agreed / total) * 0.2
        
        if agreed >= self.CONSENSUS_STRONG and confidence >= 0.85:
            return "BAN", confidence, agreed
        elif agreed >= self.CONSENSUS_MODERATE and confidence >= 0.75:
            return "MUTE", confidence, agreed
        elif agreed >= 2 and confidence >= 0.6:
            return "FLAG", confidence, agreed
        return "SKIP", confidence, agreed
    
    def semantic_score(self, text: str) -> float:
        if not text or len(text) < 5 or self.spam_embeddings is None:
            return 0.0
        emb = self.embedder.encode([text])
        sims = cosine_similarity(emb, self.spam_embeddings)[0]
        max_sim = float(np.max(sims))
        
        if self.ham_examples and len(self.ham_examples) > 0:
            ham_emb = self.embedder.encode([e.text for e in self.ham_examples])
            ham_sims = cosine_similarity(emb, ham_emb)[0]
            if float(np.max(ham_sims)) > 0.85:
                max_sim *= 0.3
        
        if max_sim > 0.90: return 0.95
        elif max_sim > 0.82: return 0.80
        elif max_sim > 0.72: return 0.55
        return 0.0
    
    def ban_user(self, user_id: int):
        self.banned_users.add(user_id)
        self._save(self.banned_users, BANNED_FILE)
    
    def is_banned(self, user_id: int) -> bool:
        return user_id in self.banned_users
    
    def add_whitelist(self, user_id: int, chat_id: int):
        self.whitelist.add((user_id, chat_id))
        self._save(self.whitelist, WHITELIST_FILE)
    
    def is_whitelisted(self, user_id: int, chat_id: int) -> bool:
        return (user_id, chat_id) in self.whitelist
    
    def stats(self):
        return {
            "spam": len(self.spam_examples),
            "ham": len(self.ham_examples),
            "banned": len(self.banned_users),
            "whitelist": len(self.whitelist),
        }
