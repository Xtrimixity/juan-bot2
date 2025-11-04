import os
import asyncio
import logging
import discord
from discord.ext import commands
from huggingface_hub import InferenceClient

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("discord_bot")

# Environment variables
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "gpt2")

if not DISCORD_TOKEN:
    raise RuntimeError("DISCORD_TOKEN not set in environment")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN not set in environment")

# Hugging Face client
hf = InferenceClient(token=HF_TOKEN)

# Discord setup
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

CHANNEL_COOLDOWN = 5  # seconds per channel
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
            result = hf.text_generation(model=HF_MODEL, inputs=prompt, max_new_tokens=200)
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


@bot.event
async def on_ready():
    logger.info(f"Bot connected as {bot.user} (id: {bot.user.id})")


@bot.command(name="ask", help="Ask the AI something. Usage: !ask <question>")
@commands.cooldown(1, 10.0, commands.BucketType.user)
async def ask(ctx, *, question: str):
    if not can_call_hf(ctx.channel.id):
        await ctx.reply("⏳ Please wait a few seconds before asking again.")
        return

    msg = await ctx.reply("💭 Thinking...")
    response = await query_hf(question)
    await msg.edit(content=response[:1900])


@bot.event
async def on_message(message: discord.Message):
    # Ignore the bot’s own messages
    if message.author == bot.user:
        return

    # If bot is mentioned, respond using Hugging Face
    if bot.user.mentioned_in(message) and not message.mention_everyone:
        if not can_call_hf(message.channel.id):
            await message.reply("⏳ Please wait a few seconds before I can respond again.")
            return

        async with message.channel.typing():
            prompt = f"{message.author.name} said: {message.content}\n\nBot reply:"
            response = await query_hf(prompt)

        await message.reply(response[:1900])

    # Process commands like !ask
    await bot.process_commands(message)


async def start_bot():
    try:
        await bot.start(DISCORD_TOKEN)
    except KeyboardInterrupt:
        await bot.close()


if __name__ == "__main__":
    asyncio.run(start_bot())
