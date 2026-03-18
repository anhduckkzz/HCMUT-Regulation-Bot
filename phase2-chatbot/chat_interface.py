import datetime
import io
import logging
import os
import traceback
import typing
import ast
import re

import aiohttp
import discord
from discord.ext import commands
from dotenv import load_dotenv

try:
    from rag_engine_langchain import generate_response
except ImportError:
    generate_response = lambda x: "RAG engine not available"


class PingCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    def _unescape_text(self, text: str) -> str:
        """Convert common escaped characters to display-friendly text."""
        # Handle double-escaped sequences first, then single-escaped ones.
        normalized = text.replace("\\\\n", "\n").replace("\\\\t", "\t")
        normalized = normalized.replace("\\n", "\n").replace("\\t", "\t")
        normalized = normalized.replace("\\'", "'").replace('\\"', '"')
        return normalized

    def _extract_text_from_stringified_payload(self, payload_text: str) -> typing.Optional[str]:
        """Extract `text` field from payload-like strings when full parsing fails."""
        # Preferred shape: {..., 'text': '...', 'extras': {...}}
        pattern_with_extras = re.search(
            r"['\"]text['\"]\s*:\s*(['\"])(.*?)\1\s*,\s*['\"]extras['\"]\s*:",
            payload_text,
            flags=re.DOTALL,
        )
        if pattern_with_extras:
            return self._unescape_text(pattern_with_extras.group(2).strip())

        # Fallback shape without extras.
        pattern_text_only = re.search(
            r"['\"]text['\"]\s*:\s*(['\"])(.*?)\1\s*}\s*$",
            payload_text,
            flags=re.DOTALL,
        )
        if pattern_text_only:
            return self._unescape_text(pattern_text_only.group(2).strip())

        return None

    def _extract_text_response(self, response: typing.Any) -> str:
        """Extract text from response, handling various response formats.
        
        Args:
            response: Response from generate_response (could be str, dict, or other types)
            
        Returns:
            Extracted response text as string
        """
        parsed_response: typing.Any = response

        # Handle dict-like response directly.
        if isinstance(parsed_response, dict):
            text_value = parsed_response.get("text")
            if isinstance(text_value, str):
                return self._unescape_text(text_value)
            return str(text_value or parsed_response)

        # Handle list-based response blocks.
        if isinstance(parsed_response, list):
            text_parts = []
            for item in parsed_response:
                if isinstance(item, dict):
                    text_parts.append(self._unescape_text(str(item.get("text", "")).strip()))
                else:
                    text_parts.append(self._unescape_text(str(item).strip()))
            return "\n".join(part for part in text_parts if part)

        # Convert to string first for text-based parsing.
        response_text = str(parsed_response).strip()

        # Sometimes SDK output is stringified dict with single quotes.
        if response_text.startswith("{") and response_text.endswith("}"):
            parsed_dict: typing.Optional[dict] = None

            try:
                maybe_dict = ast.literal_eval(response_text)
                if isinstance(maybe_dict, dict):
                    parsed_dict = maybe_dict
            except (ValueError, SyntaxError):
                parsed_dict = None

            if parsed_dict is not None:
                text_value = parsed_dict.get("text")
                if isinstance(text_value, str):
                    return self._unescape_text(text_value)
                return str(text_value or parsed_dict)

        # Last-resort extraction for payload-looking strings.
        fallback_text = self._extract_text_from_stringified_payload(response_text)
        if fallback_text:
            return fallback_text

        return self._unescape_text(response_text)

    def _parse_response(self, response: str) -> typing.Tuple[str, str]:
        """Parse the RAG engine response to separate content and sources.
        
        Args:
            response: Full response from generate_response()
            
        Returns:
            Tuple of (content, sources_text)
        """
        # Split by source marker, allowing markdown and plain-text variants.
        source_split_pattern = re.compile(
            r"\n\s*(?:\*\*)?\s*(?:📚\s*)?Nguồn tham khảo\s*:?\s*(?:\*\*)?\s*\n",
            flags=re.IGNORECASE,
        )
        parts = source_split_pattern.split(response, maxsplit=1)
        content = parts[0].strip()
        sources_text = ""
        if len(parts) > 1:
            sources_text = self._format_sources(parts[1])

        return content, sources_text

    def _format_sources(self, raw_sources: str) -> str:
        """Format source list into clean, deduplicated lines for Discord field."""
        seen = set()
        cleaned_sources = []

        for line in self._unescape_text(raw_sources).splitlines():
            line = line.strip()
            if not line:
                continue

            # Remove markdown bullets and numbering.
            line = re.sub(r"^[-*•\d.)\s]+", "", line).strip()
            if not line:
                continue

            if line not in seen:
                seen.add(line)
                cleaned_sources.append(line)

        return "\n".join(cleaned_sources)

    def _build_full_response_text(self, content: str, sources_text: str, message: str) -> str:
        """Build full plain-text response for attachment when embed limits are exceeded."""
        parts = ["🤖 AI TRẢ LỜI:", "", content.strip()]

        if sources_text:
            parts.extend(["", "📚 Nguồn tham khảo", sources_text.strip()])

        parts.extend(["", f"Câu hỏi: {message}"])
        return "\n".join(parts)

    def _build_sources_message(self, sources_text: str) -> str:
        """Build plain-text sources block for normal Discord messages."""
        if not sources_text.strip():
            return ""
        return f"📚 Nguồn tham khảo:\n{sources_text.strip()}"

    def _chunk_text(self, text: str, max_len: int = 1900) -> typing.List[str]:
        """Split text into Discord-safe chunks while preserving paragraph boundaries."""
        if len(text) <= max_len:
            return [text]

        chunks: typing.List[str] = []
        current = ""

        for paragraph in text.split("\n\n"):
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            candidate = paragraph if not current else f"{current}\n\n{paragraph}"
            if len(candidate) <= max_len:
                current = candidate
                continue

            if current:
                chunks.append(current)
                current = ""

            # Fallback when a single paragraph exceeds the limit.
            while len(paragraph) > max_len:
                chunks.append(paragraph[:max_len])
                paragraph = paragraph[max_len:]

            current = paragraph

        if current:
            chunks.append(current)

        return chunks

    @commands.command()
    async def ask(self, ctx, *, message: str = "Pong!"):
        """Responds with a custom message and latency."""
        response = generate_response(message)
        
        # Extract text if response is wrapped in metadata
        response = self._extract_text_response(response)
        
        # Parse response into content and sources
        content, sources_text = self._parse_response(response)

        # Keep full text for attachment when embed limits are exceeded.
        full_response_text = self._build_full_response_text(content, sources_text, message)
        sources_message = self._build_sources_message(sources_text)

        # Send main answer as normal message content (no embed description).
        answer_text = f"🤖 ĐANG TRẢ LỜI...\n\n{content}".strip()
        answer_chunks = self._chunk_text(answer_text, max_len=1900)
        source_chunks = self._chunk_text(sources_message, max_len=1900) if sources_message else []
        question_text = f"Câu hỏi: \"{message}\""

        need_attachment = (
            len(answer_chunks) > 4
            or len(source_chunks) > 3
            or len(full_response_text) > 5500
        )

        for chunk in answer_chunks:
            await ctx.send(chunk)

        for chunk in source_chunks:
            await ctx.send(chunk)

        await ctx.send(question_text)

        if need_attachment:
            response_file = discord.File(
                io.BytesIO(full_response_text.encode("utf-8")),
                filename="full_rag_response.txt",
            )
            await ctx.send(
                content="Mình gửi kèm file để bạn xem đầy đủ toàn bộ phản hồi, không cắt bớt nội dung.",
                file=response_file,
            )


class CustomBot(commands.Bot):
    client: aiohttp.ClientSession
    _uptime: datetime.datetime = datetime.datetime.now(datetime.UTC)

    def __init__(self, prefix: str, *args: typing.Any, **kwargs: typing.Any) -> None:
        intents = discord.Intents.default()
        intents.members = True
        intents.message_content = True
        super().__init__(*args, **kwargs, command_prefix=commands.when_mentioned_or(prefix), intents=intents)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.synced = False

    async def on_error(self, event_method: str, *args: typing.Any, **kwargs: typing.Any) -> None:
        self.logger.error(f"An error occurred in {event_method}.\n{traceback.format_exc()}")

    async def on_ready(self) -> None:
        self.logger.info(f"Logged in as {self.user} ({self.user.id})")

    async def setup_hook(self) -> None:
        self.client = aiohttp.ClientSession()
        # Add PingCog directly
        await self.add_cog(PingCog(self))
        self.logger.info("Loaded cog: PingCog")
        if not self.synced:
            await self.tree.sync()
            self.synced = not self.synced
            self.logger.info("Synced command tree")

    async def close(self) -> None:
        await super().close()
        await self.client.close()

    def run(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        load_dotenv()
        try:
            super().run(str(os.getenv("DISCORD_BOT_TOKEN")), *args, **kwargs)
        except (discord.LoginFailure, KeyboardInterrupt):
            self.logger.info("Exiting...")
            exit()

    @property
    def user(self) -> discord.ClientUser:
        assert super().user, "Bot is not ready yet"
        return typing.cast(discord.ClientUser, super().user)

    @property
    def uptime(self) -> datetime.timedelta:
        return datetime.datetime.now(datetime.UTC) - self._uptime


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    bot = CustomBot(prefix="!")
    bot.run()


if __name__ == "__main__":
    main()