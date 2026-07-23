import asyncio
import os

from dotenv import load_dotenv
from telegram import Bot

load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

async def main():
    print("Loading token...")

    if not TOKEN:
        print("❌ Telegram token not found in .env")
        return

    bot = Bot(token=TOKEN)

    me = await bot.get_me()

    print("✅ Connected Successfully")
    print(f"Bot Name : {me.first_name}")
    print(f"Username : @{me.username}")

asyncio.run(main())