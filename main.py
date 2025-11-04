import os
import asyncio
import logging
import discord
from discord.ext import commands
import httpx

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("discord_bot")

# -------------------------
# Environment variables
# -------------------------
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "gpt2")  # Replace with your preferred Groq model

if not DISCORD_TOKEN:
    raise RuntimeError("DISCORD_TOKEN not set")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set")

# -------------------------
# Discord setup
# -------------------------
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# Cooldown per channel
CHANNEL_COOLDOWN = 5
_last_called = {}

def can_call_ai(channel_id: int) -> bool:
    import time
    last = _last_called.get(channel_id, 0)
    if time.time() - last < CHANNEL_COOLDOWN:
        return False
    _last_called[channel_id] = time.time()
    return True

# -------------------------
# Groq API call
# -------------------------
async def query_groq(prompt: str) -> str:
    async with httpx.AsyncClient() as client:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": GROQ_MODEL,
            "prompt": prompt,
            "max_output_tokens": 200
        }
        try:
            resp = await client.post("https://api.groq.ai/v1/generate", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            # Groq API usually returns something like data["text"] or data["outputs"][0]["content"]
            if "outputs" in data and data["outputs"]:
                return data["outputs"][0].get("content", "")
            elif "text" in data:
                return data["text"]
            return str(data)
        except Exception as e:
            logger.exception("Groq API error")
            return f"[Error from Groq] {e}"

# -------------------------
# Discord events
# -------------------------
@bot.event
async def on_ready():
    logger.info(f"Bot connected as {bot.user} (id: {bot.user.id})")

@bot.event
async def on_message(message: discord.Message):
    if message.author == bot.user:
        return

    if bot.user.mentioned_in(message) and not message.mention_everyone:
        if not can_call_ai(message.channel.id):
            await message.reply("⏳ Please wait a few seconds before I can reply again.")
            return

        async with message.channel.typing():
            system_prompt = (
                "You are Juan, a friendly adult male who believes he is human. "
                "Respond naturally and politely in the tone of a person named Juan."
            )
            user_prompt = message.content.replace(f"<@{bot.user.id}>", "").strip()
            full_prompt = f"{system_prompt}\n\nUser said: {user_prompt}\nJuan replies:"

            response = await query_groq(full_prompt)

        await message.reply(response[:1900])

    await bot.process_commands(message)

# -------------------------
# Runner
# -------------------------
async def start_bot():
    try:
        await bot.start(DISCORD_TOKEN)
    except KeyboardInterrupt:
        await bot.close()

if __name__ == "__main__":
    asyncio.run(start_bot())
