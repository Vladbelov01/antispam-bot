"""
Хранение базы спама в Google Sheets вместо локальных файлов.
Бесплатно, не пропадает при перезапуске Render.
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Optional

import gspread
from oauth2client.service_account import ServiceAccountCredentials
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# ========== НАСТРОЙКИ ==========

# ID вашей Google таблицы (из URL)
SHEET_ID = os.getenv("GOOGLE_SHEET_ID", "")

# Email сервисного аккаунта (из JSON-ключа)
SERVICE_EMAIL = os.getenv("GOOGLE_SERVICE_EMAIL", "")

# Приватный ключ (весь текст из JSON, в одну строку через \n)
PRIVATE_KEY = os.getenv("GOOGLE_PRIVATE_KEY", "").replace('\\n', '\n')

# Названия листов
SPAM_SHEET = "SpamBase"
HAM_SHEET = "HamBase"
BANNED_SHEET = "Banned"
WHITELIST_SHEET = "Whitelist"


class SheetsStorage:
    """Всё хранится в Google Sheets, ничего не теряется при перезапуске"""
    
    def __init__(self):
        self.client = None
        self.spam_sheet = None
        self.ham_sheet = None
        self.banned_sheet = None
        self.whitelist_sheet = None
        
        self._connect()
        self._ensure_sheets()
    
    def _connect(self):
        """Подключается к Google Sheets через сервисный аккаунт"""
        if not all([SHEET_ID, SERVICE_EMAIL, PRIVATE_KEY]):
            raise ValueError("❌ Не заданы переменные Google Sheets. Проверьте GOOGLE_SHEET_ID, GOOGLE_SERVICE_EMAIL, GOOGLE_PRIVATE_KEY")
        
        # Создаём временный JSON-файл для авторизации
        creds_data = {
            "type": "service_account",
            "project_id": "antispam-bot",
            "private_key_id": "key",
            "private_key": PRIVATE_KEY,
            "client_email": SERVICE_EMAIL,
            "client_id": "123",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
        
        with open("/tmp/credentials.json", "w") as f:
            json.dump(creds_data, f)
        
        scope = [
            'https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive'
        ]
        
        creds = ServiceAccountCredentials.from_json_keyfile_name("/tmp/credentials.json", scope)
        self.client = gspread.authorize(creds)
    
    def _ensure_sheets(self):
        """Создаёт листы если их нет"""
        try:
            spreadsheet = self.client.open_by_key(SHEET_ID)
        except gspread.SpreadsheetNotFound:
            raise ValueError(f"❌ Таблица {SHEET_ID} не найдена. Проверьте доступ.")
        
        existing = [ws.title for ws in spreadsheet.worksheets()]
        
        for name in [SPAM_SHEET, HAM_SHEET, BANNED_SHEET, WHITELIST_SHEET]:
            if name not in existing:
                spreadsheet.add_worksheet(title=name, rows=1000, cols=10)
                # Добавляем заголовки
                ws = spreadsheet.worksheet(name)
                if name == SPAM_SHEET:
                    ws.append_row(["text", "hash", "timestamp", "confidence", "detectors_agreed", "user_id", "chat_id"])
                elif name == HAM_SHEET:
                    ws.append_row(["text", "hash", "timestamp", "user_id", "chat_id"])
                elif name == BANNED_SHEET:
                    ws.append_row(["user_id", "timestamp"])
                elif name == WHITELIST_SHEET:
                    ws.append_row(["user_id", "chat_id", "timestamp"])
        
        self.spam_sheet = spreadsheet.worksheet(SPAM_SHEET)
        self.ham_sheet = spreadsheet.worksheet(HAM_SHEET)
        self.banned_sheet = spreadsheet.worksheet(BANNED_SHEET)
        self.whitelist_sheet = spreadsheet.worksheet(WHITELIST_SHEET)
    
    # ========== SPAM ==========
    
    def get_spam_texts(self) -> List[str]:
        """Возвращает все тексты спама из таблицы"""
        try:
            records = self.spam_sheet.get_all_records()
            return [r["text"] for r in records if r.get("text")]
        except:
            return []
    
    def add_spam(self, text: str, text_hash: str, confidence: float,
                 detectors_agreed: int, detector_scores: Dict[str, float],
                 user_id: Optional[int] = None, chat_id: Optional[int] = None) -> bool:
        """Добавляет спам в таблицу"""
        if not text or len(text.strip()) < 5:
            return False
        
        # Проверяем дубликат по хешу
        existing = self.spam_sheet.get_all_records()
        for row in existing:
            if row.get("hash") == text_hash:
                return False
        
        self.spam_sheet.append_row([
            text.strip(),
            text_hash,
            datetime.now().isoformat(),
            str(confidence),
            str(detectors_agreed),
            str(user_id or ""),
            str(chat_id or ""),
        ])
        return True
    
    # ========== HAM ==========
    
    def get_ham_texts(self) -> List[str]:
        try:
            records = self.ham_sheet.get_all_records()
            return [r["text"] for r in records if r.get("text")]
        except:
            return []
    
    def add_ham(self, text: str, text_hash: str, user_id: Optional[int] = None, chat_id: Optional[int] = None):
        if not text or len(text.strip()) < 5:
            return
        
        existing = self.ham_sheet.get_all_records()
        for row in existing:
            if row.get("hash") == text_hash:
                return
        
        self.ham_sheet.append_row([
            text.strip(),
            text_hash,
            datetime.now().isoformat(),
            str(user_id or ""),
            str(chat_id or ""),
        ])
    
    # ========== BANNED ==========
    
    def get_banned(self) -> set:
        try:
            records = self.banned_sheet.get_all_records()
            return {int(r["user_id"]) for r in records if r.get("user_id")}
        except:
            return set()
    
    def ban_user(self, user_id: int):
        banned = self.get_banned()
        if user_id not in banned:
            self.banned_sheet.append_row([str(user_id), datetime.now().isoformat()])
    
    def unban_user(self, user_id: int):
        # Удаляем строку с user_id (gspread не поддерживает удаление напрямую, очищаем и перезаписываем)
        records = self.banned_sheet.get_all_records()
        new_records = [r for r in records if str(r.get("user_id")) != str(user_id)]
        
        self.banned_sheet.clear()
        self.banned_sheet.append_row(["user_id", "timestamp"])
        for r in new_records:
            self.banned_sheet.append_row([str(r["user_id"]), r.get("timestamp", "")])
    
    # ========== WHITELIST ==========
    
    def get_whitelist(self) -> set:
        try:
            records = self.whitelist_sheet.get_all_records()
            return {(int(r["user_id"]), int(r["chat_id"])) for r in records if r.get("user_id") and r.get("chat_id")}
        except:
            return set()
    
    def add_whitelist(self, user_id: int, chat_id: int):
        wl = self.get_whitelist()
        if (user_id, chat_id) not in wl:
            self.whitelist_sheet.append_row([str(user_id), str(chat_id), datetime.now().isoformat()])
