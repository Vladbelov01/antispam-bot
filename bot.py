import asyncio
import os
import logging
from datetime import datetime, timedelta
from collections import defaultdict

from aiogram import Bot, Dispatcher, types
from aiogram.types import ChatMemberUpdated
from aiogram.enums import ChatMemberStatus

from config import BOT_TOKEN, BOT_MODE, AUTO_BAN_THRESHOLD, AUTO_MUTE_THRESHOLD, AUTO_DELETE_THRESHOLD, WHITELIST_AFTER_MESSAGES
from brain import Brain
from detectors import Detectors


logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("bot")

brain = Brain()
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

user_messages = defaultdict(list)
user_join_time = {}
user_msg_count = defaultdict(int)


@dp.chat_member()
async def on_join(event: ChatMemberUpdated):
    if event.new_chat_member and event.new_chat_member.user:
        key = (event.new_chat_member.user.id, event.chat.id)
        user_join_time[key] = datetime.now()


@dp.message()
async def handle(message: types.Message):
    if not message.from_user:
        return
    
    uid = message.from_user.id
    cid = message.chat.id
    key = (uid, cid)
    
    if message.from_user.is_bot:
        return
    
    try:
        m = await bot.get_chat_member(cid, uid)
        if m.status in (ChatMemberStatus.ADMINISTRATOR, ChatMemberStatus.CREATOR):
            return
    except:
        pass
    
    if brain.is_whitelisted(uid, cid):
        return
    
    user_msg_count[key] += 1
    if user_msg_count[key] > WHITELIST_AFTER_MESSAGES:
        brain.add_whitelist(uid, cid)
        return
    
    if brain.is_banned(uid):
        try:
            await message.chat.ban(uid)
            await message.delete()
        except:
            pass
        return
    
    text = message.text or message.caption or ""
    if not text:
        return
    
    results = Detectors.run_all(
        name=message.from_user.full_name or "",
        username=message.from_user.username,
        text=text,
        history=user_messages.get(key, []),
        join_time=user_join_time.get(key)
    )
    
    decision, confidence, agreed = brain.consensus(results)
    
    semantic = brain.semantic_score(text)
    if semantic > 0.5:
        results["semantic"] = semantic
        decision, confidence, agreed = brain.consensus(results)
    
    user_messages[key].append({'time': datetime.now(), 'text': text, 'score': confidence})
    if len(user_messages[key]) > 20:
        user_messages[key] = user_messages[key][-20:]
    
    added = False
    if decision in ("BAN", "MUTE") and confidence >= 0.80:
        added = brain.add_spam(text, confidence, agreed, results, uid, cid)
    
    action = "LOGGED"
    
    if BOT_MODE == "ACTIVE":
        try:
            if decision == "BAN" and confidence >= AUTO_BAN_THRESHOLD:
                await message.chat.ban(uid)
                await message.delete()
                brain.ban_user(uid)
                action = "AUTO_BAN"
                logger.warning(f"BAN: {message.from_user.full_name} | {confidence:.2f}")
                
            elif decision == "MUTE" and confidence >= AUTO_MUTE_THRESHOLD:
                until = datetime.now() + timedelta(hours=24)
                await message.chat.restrict(uid, until_date=until, can_send_messages=False)
                await message.delete()
                action = "AUTO_MUTE"
                logger.warning(f"MUTE: {message.from_user.full_name} | {confidence:.2f}")
                
            elif decision == "FLAG" and confidence >= AUTO_DELETE_THRESHOLD:
                await message.delete()
                action = "AUTO_DELETE"
                
        except Exception as e:
            logger.error(f"Action error: {e}")
    
    if confidence > 0.5:
        mark = " [+BASE]" if added else ""
        logger.info(f"{decision}: {message.from_user.full_name} | {confidence:.2f} | {agreed}/6{mark}")


@dp.message(lambda m: m.text and m.text.startswith("/stats"))
async def stats(message: types.Message):
    if not message.from_user:
        return
    try:
        m = await bot.get_chat_member(message.chat.id, message.from_user.id)
        if m.status not in (ChatMemberStatus.ADMINISTRATOR, ChatMemberStatus.CREATOR):
            return
    except:
        return
    
    s = brain.stats()
    await message.reply(
        f"📊 Статистика\n\n"
        f"Спам-шаблонов: {s['spam']}\n"
        f"Чистых: {s['ham']}\n"
        f"Забанено: {s['banned']}\n"
        f"В белом списке: {s['whitelist']}\n"
        f"Режим: {BOT_MODE}"
    )


@dp.message(lambda m: m.text and m.text.startswith("/mode"))
async def mode_cmd(message: types.Message):
    global BOT_MODE
    if not message.from_user:
        return
    try:
        m = await bot.get_chat_member(message.chat.id, message.from_user.id)
        if m.status not in (ChatMemberStatus.ADMINISTRATOR, ChatMemberStatus.CREATOR):
            return
    except:
        return
    
    parts = message.text.split()
    if len(parts) < 2:
        await message.reply(f"Режим: {BOT_MODE}")
        return
    
    new = parts[1].upper()
    if new not in ("PASSIVE", "ACTIVE"):
        await message.reply("PASSIVE или ACTIVE")
        return
    
    BOT_MODE = new
    logger.info(f"Mode changed to: {BOT_MODE}")
    await message.reply(f"✅ Режим: {BOT_MODE}")


@dp.message(lambda m: m.text and m.text.startswith("/notspam"))
async def notspam(message: types.Message):
    if not message.from_user or not message.reply_to_message:
        return
    try:
        m = await bot.get_chat_member(message.chat.id, message.from_user.id)
        if m.status not in (ChatMemberStatus.ADMINISTRATOR, ChatMemberStatus.CREATOR):
            return
    except:
        return
    
    target = message.reply_to_message
    if not target.from_user:
        return
    
    txt = target.text or target.caption or ""
    
    if brain.is_banned(target.from_user.id):
        try:
            await message.chat.unban(target.from_user.id)
            brain.banned_users.discard(target.from_user.id)
            brain._save(brain.banned_users, "data/banned_users.json")
        except:
            pass
    
    if txt:
        brain.add_ham(txt, target.from_user.id, message.chat.id)
    
    brain.add_whitelist(target.from_user.id, message.chat.id)
    await message.reply("✅ Добавлено в белый список")


async def main():
    logger.info("🚀 Bot started")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
