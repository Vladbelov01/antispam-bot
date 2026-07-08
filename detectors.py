import re
from typing import Optional, Dict, List
from datetime import datetime, timedelta

import pymorphy3
morph = pymorphy3.MorphAnalyzer()

ADJ = {'сладкий', 'нежный', 'знойный', 'чудесный', 'весенний', 'горячий', 'мягкий', 'страстный'}
NOUNS = {'мелодия', 'бандитка', 'шалунья', 'крошка', 'кошечка', 'зайка', 'фея', 'луна', 'кармелька'}

SEX_TRIGGERS = {'приват': 0.3, 'интим': 0.4, 'вписка': 0.3, 'горячий': 0.15, 'желание': 0.15, 'одиночество': 0.25,
                'спасти': 0.2, 'раздевать': 0.3, 'снимать': 0.2, 'поболтать': 0.15, 'салют': 0.1, 'приветик': 0.1}

JOB_TRIGGERS = {'удалённо': 0.15, 'удалёнка': 0.15, 'дополнительный доход': 0.2, 'пассивный доход': 0.3,
                'без вложений': 0.2, 'без опыта': 0.15, 'обучим': 0.15, 'ежедневная оплата': 0.2,
                'работа из дома': 0.15, '2-3 часа': 0.15, 'от 500': 0.15, 'доход с первого дня': 0.25,
                'партнёрская программа': 0.25, 'крипта': 0.15, 'нужен человек': 0.15, 'ищу человека': 0.15,
                'пишите в лс': 0.2, 'пиши в лс': 0.2, 'в личных сообщениях': 0.2, 'подробности в лс': 0.25}


class Detectors:
    @staticmethod
    def nickname(name: str) -> float:
        if not name or len(name) < 3:
            return 0.0
        clean = re.sub(r'[^\w\s]', '', name).strip().lower()
        words = clean.split()
        if len(words) != 2:
            return 0.0
        p1, p2 = morph.parse(words[0]), morph.parse(words[1])
        if not p1 or not p2:
            return 0.0
        score = 0.0
        if p1[0].tag.POS in ('ADJF', 'ADJS') and p2[0].tag.POS == 'NOUN':
            score += 0.5
            if 'femn' in str(p2[0].tag):
                score += 0.2
        if p1[0].normal_form in ADJ:
            score += 0.1
        if p2[0].normal_form in NOUNS:
            score += 0.1
        return min(score, 1.0)
    
    @staticmethod
    def sex_text(text: str) -> float:
        if not text:
            return 0.0
        t = text.lower()
        score = sum(w for tr, w in SEX_TRIGGERS.items() if tr in t)
        patterns = [r'привет.*я как.*кофе', r'жду твоего.*внимания', r'раздеваешь.*взглядом',
                    r'потерялись.*желаниях', r'спасти.*одиночества', r'налил[а].*бокал']
        for p in patterns:
            if re.search(p, t):
                score += 0.4
        emojis = len(re.findall(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF]', text))
        if emojis >= 2:
            score += 0.1
        return min(score, 1.0)
    
    @staticmethod
    def job_text(text: str) -> float:
        if not text:
            return 0.0
        t = text.lower()
        score = sum(w for tr, w in JOB_TRIGGERS.items() if tr in t)
        patterns = [r'ищу\s+\d+[-\s]?\d+\s+человек', r'от\s+\d+\s*\$', r'\d+[-\s]?\d+\s*часа?\s*в\s*день',
                    r'доход\s*с\s*первого\s*дня', r'пиши\s*в\s*лс', r'партн[её]рская\s*программа']
        for p in patterns:
            if re.search(p, t):
                score += 0.3
        return min(score, 1.0)
    
    @staticmethod
    def username(name: Optional[str]) -> float:
        if not name:
            return 0.0
        if re.match(r'^[a-z]+_[a-z]+_\d{4,}$', name):
            return 0.5
        if re.match(r'^[a-z]+_\d{4,}$', name):
            return 0.4
        if re.match(r'^id\d+$', name):
            return 0.3
        return 0.0
    
    @staticmethod
    def links(text: str) -> float:
        if not text:
            return 0.0
        score = 0.0
        t = text.lower()
        if re.search(r'(?:t\.me|telegram\.me)/[a-zA-Z0-9_]+', t):
            score += 0.2
        if re.search(r'https?://(?!t\.me|telegram\.me)', t):
            score += 0.3
        if score > 0 and any(w in t for w in ['пиши', 'переходи', 'подпишись']):
            score += 0.3
        return min(score, 1.0)
    
    @staticmethod
    def behavior(history: List[dict], join_time: Optional[datetime], current_score: float) -> float:
        if not history:
            return 0.0
        score = 0.0
        if len(history) >= 3:
            recent = history[-3:]
            span = (recent[-1]['time'] - recent[0]['time']).total_seconds()
            if span < 300 and sum(1 for m in recent if m.get('score', 0) > 0.4) >= 2:
                score += 0.4
        if join_time and len(history) <= 2:
            if (history[0]['time'] - join_time) < timedelta(minutes=2) and current_score > 0.4:
                score += 0.3
        if len(history) >= 4:
            if sum(1 for m in history if m.get('score', 0) > 0.3) / len(history) > 0.8:
                score += 0.3
        return min(score, 1.0)
    
    @classmethod
    def run_all(cls, name: str, username: Optional[str], text: str,
                history: List[dict], join_time: Optional[datetime]) -> Dict[str, float]:
        sex = cls.sex_text(text)
        job = cls.job_text(text)
        text_score = max(sex, job)
        return {
            "nickname": cls.nickname(name),
            "text_patterns": text_score,
            "username": cls.username(username),
            "links": cls.links(text),
            "behavior": cls.behavior(history, join_time, text_score),
        }
