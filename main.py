import os
import asyncio
import logging
import discord
from discord.ext import commands
from huggingface_hub import InferenceClient

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("discord_bot")

# -------------------------
# Env variables
# -------------------------
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "tiiuae/falcon-7b-instruct")  # updated model

if not DISCORD_TOKEN:
    raise RuntimeError("DISCORD_TOKEN not set")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN not set")

# -------------------------
# Hugging Face client
# -------------------------
hf = InferenceClient(token=HF_TOKEN)

# -------------------------
# Discord setup
# -------------------------
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# Cooldown per channel
CHANNEL_COOLDOWN = 5
_last_called = {}

def can_call_hf(channel_id: int) -> bool:
    import time
    last = _last_called.get(channel_id, 0)
    if time.time() - last < CHANNEL_COOLDOWN:
        return False
    _last_called[channel_id] = time.time()
    return True

async def query_hf(prompt: str) -> str:
    loop = asyncio.get_event_loop()

    def sync_call():
        try:
            # Hugging Face text_generation: positional prompt argument
            result = hf.text_generation(prompt, model=HF_MODEL, max_new_tokens=200)
            if isinstance(result, list) and result:
                item = result[0]
                if isinstance(item, dict):
                    return item.get("generated_text") or str(item)
                return str(item)
            return str(result)
        except StopIteration:
            return f"[Hugging Face Error] Model '{HF_MODEL}' is not available for inference. Use a model with the Inference API enabled."
        except Exception as e:
            logger.exception("Hugging Face error")
            return f"[Error from Hugging Face] {e}"

    return await loop.run_in_executor(None, sync_call)

# -------------------------
# Events
# -------------------------
@bot.event
async def on_ready():
    logger.info(f"Bot connected as {bot.user} (id: {bot.user.id})")

@bot.event
async def on_message(message: discord.Message):
    if message.author == bot.user:
        return

    if bot.user.mentioned_in(message) and not message.mention_everyone:
        if not can_call_hf(message.channel.id):
            await message.reply("⏳ Please wait a few seconds before I can reply again.")
            return

        async with message.channel.typing():
            system_prompt = (
                "You are Juan, a friendly adult male who believes he is human. "
                "Respond naturally and politely in the tone of a person named Juan."
            )

            user_prompt = message.content.replace(f"<@{bot.user.id}>", "").strip()
            prompt = f"{system_prompt}\n\nUser said: {user_prompt}\nJuan replies:"

            response = await query_hf(prompt)

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
