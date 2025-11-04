# bot.py
import os
import asyncio
import logging
from typing import Optional

import discord
from discord.ext import commands
from huggingface_hub import InferenceClient, inference

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("discord_bot")

# Environment
DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN")
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_MODEL = os.environ.get("HF_MODEL", "gpt2")  # change to preferred model

if not DISCORD_TOKEN:
    raise RuntimeError("DISCORD_TOKEN not set in environment")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN not set in environment")

# Create HF Inference client
hf = InferenceClient(HF_TOKEN)

# Discord bot setup
intents = discord.Intents.default()
intents.message_content = True  # required for reading message content

bot = commands.Bot(command_prefix="!", intents=intents, help_command=commands.DefaultHelpCommand())

# Simple rate-limiter per-channel to avoid spamming the HF API
CHANNEL_COOLDOWN = 5  # seconds between allowed HF queries per channel
_last_called = {}

def can_call_hf(channel_id: int) -> bool:
    import time
    last = _last_called.get(channel_id, 0)
    if time.time() - last < CHANNEL_COOLDOWN:
        return False
    _last_called[channel_id] = time.time()
    return True

async def query_hf(prompt: str, model: Optional[str] = None) -> str:
    """
    Uses Hugging Face InferenceClient to get a text response.
    This function uses the general inference endpoint and will adapt to model pipeline type.
    """
    model = model or HF_MODEL
    # Wrap in thread executor if the client is blocking (hf client is synchronous but lightweight).
    # The library often supports async but using run_in_executor is safe.
    loop = asyncio.get_event_loop()
    def sync_call():
        # InferenceClient.text_generation returns a dict-like or string depending on model
        try:
            # Use the "text generation" API style; some models may need different kwargs
            res = hf.text_generation(model=model, inputs=prompt, max_new_tokens=256)
            # res may be a list of dicts or string
            if isinstance(res, list) and res:
                # many HF text generation return list of dicts with 'generated_text'
                first = res[0]
                if isinstance(first, dict):
                    return first.get("generated_text") or str(first)
                return str(first)
            return str(res)
        except Exception as e:
            logger.exception("Hugging Face inference error")
            return f"[HF Error] {e}"

    out = await loop.run_in_executor(None, sync_call)
    return out

@bot.event
async def on_ready():
    logger.info(f"Bot connected as {bot.user} (id: {bot.user.id})")

@bot.command(name="ask", help="Ask the AI something. Usage: !ask <question>")
@commands.cooldown(1, 10.0, commands.BucketType.user)  # per-user cooldown
async def ask(ctx: commands.Context, *, question: str):
    if not can_call_hf(ctx.channel.id):
        await ctx.reply("Please wait a few seconds between questions in this channel.")
        return

    await ctx.defer() if hasattr(ctx, "defer") else None

    waiting = await ctx.reply("Thinking... (querying Hugging Face)")
    try:
        response = await query_hf(question)
        # Trim too long messages: Discord limit ~2000 chars
        if len(response) > 1900:
            response = response[:1900] + "\n\n...[truncated]"
        await waiting.edit(content=response)
    except Exception as e:
        logger.exception("Failed to answer")
        await waiting.edit(content=f"Error: {e}")

@bot.command(name="ping", help="Check bot latency")
async def ping(ctx: commands.Context):
    await ctx.send(f"Pong! latency: {bot.latency*1000:.0f}ms")

# Graceful shutdown: ensures bot token is closed when process ends
async def start_bot():
    try:
        await bot.start(DISCORD_TOKEN)
    except KeyboardInterrupt:
        await bot.close()

if __name__ == "__main__":
    # Run the bot forever; Render background worker will run this process.
    try:
        asyncio.run(start_bot())
    except Exception as e:
        logger.exception("Bot crashed")
        raise
