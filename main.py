import os
import asyncio
import logging
import discord
from discord.ext import commands
from huggingface_hub import InferenceClient

# ---------------------------------------------------------
# Logging setup
# ---------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("discord_bot")

# ---------------------------------------------------------
# Environment variables
# ---------------------------------------------------------
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "gpt2")

if not DISCORD_TOKEN:
    raise RuntimeError("DISCORD_TOKEN not set")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN not set")

# ---------------------------------------------------------
# Hugging Face client
# ---------------------------------------------------------
hf = InferenceClient(token=HF_TOKEN)

# ---------------------------------------------------------
# Discord setup
# ---------------------------------------------------------
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# Basic cooldown control
CHANNEL_COOLDOWN = 5  # seconds
_last_called = {}


def can_call_hf(channel_id: int) -> bool:
    """Prevent spamming Hugging Face API per channel."""
    import time
    last = _last_called.get(channel_id, 0)
    if time.time() - last < CHANNEL_COOLDOWN:
        return False
    _last_called[channel_id] = time.time()
    return True


async def query_hf(prompt: str) -> str:
    """Send prompt to Hugging Face Inference API."""
    loop = asyncio.get_event_loop()

    def sync_call():
        try:
            # New API style: first positional arg = prompt
            result = hf.text_generation(prompt, model=HF_MODEL, max_new_tokens=200)

            # Handle different possible formats
            if isinstance(result, list) and result:
                item = result[0]
                if isinstance(item, dict):
                    return item.get("generated_text") or str(item)
                return str(item)
            return str(result)
        except Exception as e:
            logger.exception("Hugging Face error")
            return f"[Error from Hugging Face] {e}"

    return await loop.run_in_executor(None, sync_call)


# ---------------------------------------------------------
# Events
# ---------------------------------------------------------
@bot.event
async def on_ready():
    logger.info(f"Bot connected as {bot.user} (id: {bot.user.id})")


@bot.event
async def on_message(message: discord.Message):
    """Reply only when the bot is mentioned."""
    if message.author == bot.user:
        return

    if bot.user.mentioned_in(message) and not message.mention_everyone:
        if not can_call_hf(message.channel.id):
            await message.reply("⏳ Please wait a few seconds before I can reply again.")
            return

        async with message.channel.typing():
            # Give the AI its personality
            system_prompt = (
                "You are Juan, a friendly adult man who believes he is human. "
                "You enjoy chatting casually and answering questions. "
                "Respond naturally and politely in the tone of a person named Juan."
            )

            # Remove the mention text from the message
            user_prompt = message.content.replace(f"<@{bot.user.id}>", "").strip()
            prompt = f"{system_prompt}\n\nUser said: {user_prompt}\nJuan replies:"

            response = await query_hf(prompt)

        # Send trimmed message (Discord limit ~2000 chars)
        await message.reply(response[:1900])

    # Allow command processing (if you ever add commands)
    await bot.process_commands(message)


# ---------------------------------------------------------
# Runner
# ---------------------------------------------------------
async def start_bot():
    try:
        await bot.start(DISCORD_TOKEN)
    except KeyboardInterrupt:
        await bot.close()


if __name__ == "__main__":
    asyncio.run(start_bot())
``