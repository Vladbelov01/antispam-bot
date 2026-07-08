import os
import hashlib
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from sheets_storage import SheetsStorage


class Brain:
    CONSENSUS_STRONG = 4
    CONSENSUS_MODERATE = 3
    
    def __init__(self):
        print("🔄 Загрузка модели...")
        self.embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        print("✅ Модель загружена")
        
        print("🔄 Подключение к Google Sheets...")
        self.storage = SheetsStorage()
        print("✅ Google Sheets подключена")
        
        self.spam_embeddings = None
        self.ham_embeddings = None
        self._rebuild_embeddings()
    
    def _rebuild_embeddings(self):
        spam_texts = self.storage.get_spam_texts()
        if spam_texts:
            self.spam_embeddings = self.embedder.encode(spam_texts)
        else:
            self.spam_embeddings = None
        
        ham_texts = self.storage.get_ham_texts()
        if ham_texts:
            self.ham_embeddings = self.embedder.encode(ham_texts)
        else:
            self.ham_embeddings = None
        
        print(f"📊 Загружено: {len(spam_texts)} spam, {len(ham_texts)} ham")
    
    def _hash(self, text: str) -> str:
        return hashlib.md5(text.lower().strip().encode()).hexdigest()
    
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
        
        if self.ham_embeddings is not None and len(self.ham_embeddings) > 0:
            ham_sims = cosine_similarity(emb, self.ham_embeddings)[0]
            if float(np.max(ham_sims)) > 0.85:
                max_sim *= 0.3
        
        if max_sim > 0.90: return 0.95
        elif max_sim > 0.82: return 0.80
        elif max_sim > 0.72: return 0.55
        return 0.0
    
    def add_spam(self, text: str, confidence: float, agreed: int,
                 scores: Dict[str, float], user_id=None, chat_id=None) -> bool:
        h = self._hash(text)
        success = self.storage.add_spam(text, h, confidence, agreed, scores, user_id, chat_id)
        if success:
            self._rebuild_embeddings()
        return success
    
    def add_ham(self, text: str, user_id=None, chat_id=None):
        h = self._hash(text)
        self.storage.add_ham(text, h, user_id, chat_id)
    
    def ban_user(self, user_id: int):
        self.storage.ban_user(user_id)
    
    def is_banned(self, user_id: int) -> bool:
        return user_id in self.storage.get_banned()
    
    def add_whitelist(self, user_id: int, chat_id: int):
        self.storage.add_whitelist(user_id, chat_id)
    
    def is_whitelisted(self, user_id: int, chat_id: int) -> bool:
        return (user_id, chat_id) in self.storage.get_whitelist()
    
    def stats(self):
        return {
            "spam": len(self.storage.get_spam_texts()),
            "ham": len(self.storage.get_ham_texts()),
            "banned": len(self.storage.get_banned()),
            "whitelist": len(self.storage.get_whitelist()),
        }
