"""
==============================================================================
Zenemoo AI - Telegram Bot Async Job Queue Manager
==============================================================================
Manages background queue for incoming Telegram image enhancement jobs.
Prevents Telegram client timeouts and freezes by returning callback responses
immediately and processing heavy AI requests in a decoupled worker loop.
"""

import time
import io
import asyncio
from typing import Dict, Any, Optional
from telegram import Update
from telegram.ext import ContextTypes
from core.logging import logger
from clients.telegram.services.bot_api_client import bot_api_client
from shared.exceptions.telegram_exception import TelegramBackendCommunicationException


class TelegramJobQueueManager:
    """Enterprise Async Job Queue for Telegram Bot Client."""

    def __init__(self):
        self._queue: asyncio.Queue = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        self._is_running = False

    def start_worker(self, bot_app) -> None:
        """Starts the background worker loop if not already running."""
        if not self._is_running:
            self._is_running = True
            self._worker_task = asyncio.create_task(self._worker_loop(bot_app))
            logger.info("🤖 Telegram Bot Async Job Queue Worker started.")

    async def enqueue_job(
        self,
        chat_id: int,
        file_id: str,
        action_name: str,
        endpoint: str,
        status_msg_id: int,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> int:
        """Pushes new image processing job onto the queue and returns current queue position."""
        job = {
            "chat_id": chat_id,
            "file_id": file_id,
            "action_name": action_name,
            "endpoint": endpoint,
            "status_msg_id": status_msg_id,
            "context": context,
            "enqueued_at": time.time(),
        }
        await self._queue.put(job)
        return self._queue.qsize()

    async def _worker_loop(self, bot_app) -> None:
        """Background worker consuming and processing jobs sequentially."""
        while self._is_running:
            try:
                job = await self._queue.get()
                await self._process_single_job(job)
                self._queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in Telegram job queue worker loop: {e}", exc_info=True)
                await asyncio.sleep(1.0)

    async def _process_single_job(self, job: Dict[str, Any]) -> None:
        """Processes a single enqueued Telegram job."""
        chat_id = job["chat_id"]
        file_id = job["file_id"]
        action_name = job["action_name"]
        endpoint = job["endpoint"]
        status_msg_id = job["status_msg_id"]
        context = job["context"]
        start_time = time.time()

        # Update status message to processing
        try:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=status_msg_id,
                text=f"⚡ **Processing AI Engine [{action_name}]...**\n\nDownloading photo and running PyTorch model pipeline...",
                parse_mode="Markdown",
            )
        except Exception:
            pass

        try:
            import os
            from clients.telegram.config import bot_settings

            # 1. Download image bytes from Telegram servers
            tg_file = await context.bot.get_file(file_id, read_timeout=120, write_timeout=120, connect_timeout=60)
            byte_stream = io.BytesIO()
            await tg_file.download_to_memory(out=byte_stream)
            image_bytes = byte_stream.getvalue()

            # 2. Delegate to FastAPI REST API backend (with timeout protection)
            api_result = await bot_api_client.send_image_job(
                endpoint=endpoint,
                image_bytes=image_bytes,
                filename=f"tg_{file_id[:8]}.jpg",
            )

            elapsed = round(time.time() - start_time, 2)
            output_path = api_result.get("output_path")
            output_name = api_result.get("output_name")
            output_url = f"{bot_settings.BACKEND_API_URL}/outputs/{output_name}" if output_name else None

            caption = (
                f"✅ **Image Processing Complete!**\n\n"
                f"• Mode: `{action_name}`\n"
                f"• Processing Time: `{elapsed}s`\n"
                f"• Backend Status: `{api_result.get('status', 'success')}`\n"
            )
            if output_url:
                caption += f"🌐 **Direct Image Link:** `{output_url}`\n"

            # 3. Send output image back to user (with photo / document / link fallbacks)
            if output_path and os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                sent = False

                # Attempt 1: Send Photo if under 10 MB Telegram photo limit
                if file_size <= 10 * 1024 * 1024:
                    try:
                        with open(output_path, "rb") as out_f:
                            await context.bot.send_photo(
                                chat_id=chat_id,
                                photo=out_f,
                                caption=caption,
                                parse_mode="Markdown",
                                read_timeout=120,
                                write_timeout=120,
                                connect_timeout=60,
                            )
                        sent = True
                    except Exception as err_photo:
                        logger.warning(f"send_photo with Markdown failed ({err_photo}). Retrying send_photo without Markdown...")
                        try:
                            # Strip markdown backticks/asterisks for plain text fallback
                            plain_caption = caption.replace("**", "").replace("`", "")
                            with open(output_path, "rb") as out_f:
                                await context.bot.send_photo(
                                    chat_id=chat_id,
                                    photo=out_f,
                                    caption=plain_caption,
                                    parse_mode=None,
                                    read_timeout=120,
                                    write_timeout=120,
                                    connect_timeout=60,
                                )
                            sent = True
                        except Exception as err_photo_plain:
                            logger.warning(f"send_photo plain failed ({err_photo_plain}). Trying send_document...")

                # Attempt 2: Send Document if over 10 MB or send_photo failed
                if not sent:
                    try:
                        with open(output_path, "rb") as out_f:
                            await context.bot.send_document(
                                chat_id=chat_id,
                                document=out_f,
                                caption=caption,
                                parse_mode="Markdown",
                                read_timeout=120,
                                write_timeout=120,
                                connect_timeout=60,
                            )
                        sent = True
                    except Exception as err_doc:
                        logger.warning(f"send_document with Markdown failed ({err_doc}). Retrying plain send_document...")
                        try:
                            plain_caption = caption.replace("**", "").replace("`", "")
                            with open(output_path, "rb") as out_f:
                                await context.bot.send_document(
                                    chat_id=chat_id,
                                    document=out_f,
                                    caption=plain_caption,
                                    parse_mode=None,
                                    read_timeout=120,
                                    write_timeout=120,
                                    connect_timeout=60,
                                )
                            sent = True
                        except Exception as err_doc_plain:
                            logger.warning(f"send_document plain failed ({err_doc_plain}). Sending text notification with web link...")

                # Attempt 3: Direct Web Link text message fallback
                if not sent:
                    try:
                        await context.bot.send_message(
                            chat_id=chat_id,
                            text=caption,
                            parse_mode="Markdown",
                            read_timeout=120,
                            write_timeout=120,
                            connect_timeout=60,
                        )
                    except Exception:
                        plain_caption = caption.replace("**", "").replace("`", "")
                        await context.bot.send_message(
                            chat_id=chat_id,
                            text=plain_caption,
                            parse_mode=None,
                            read_timeout=120,
                            write_timeout=120,
                            connect_timeout=60,
                        )
            else:
                # Fallback send original photo with processed metadata
                try:
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text=caption,
                        parse_mode="Markdown",
                        read_timeout=120,
                        write_timeout=120,
                        connect_timeout=60,
                    )
                except Exception:
                    plain_caption = caption.replace("**", "").replace("`", "")
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text=plain_caption,
                        parse_mode=None,
                        read_timeout=120,
                        write_timeout=120,
                        connect_timeout=60,
                    )



            # Clean up temporary status message
            try:
                await context.bot.delete_message(chat_id=chat_id, message_id=status_msg_id)
            except Exception:
                pass

        except TelegramBackendCommunicationException as e:
            logger.error(f"Backend API error for job {action_name}: {e}")
            try:
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=status_msg_id,
                    text=f"❌ **Processing Failed:** {e.message}\n\nPlease try again or send a smaller image.",
                    parse_mode="Markdown",
                )
            except Exception:
                pass
        except Exception as e:
            logger.error(f"Unexpected error processing job {action_name}: {e}", exc_info=True)
            try:
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=status_msg_id,
                    text=f"❌ **Processing Error:** {str(e)}\n\nAn unexpected error occurred during processing.",
                    parse_mode="Markdown",
                )
            except Exception:
                pass


# Singleton Queue Instance
job_queue_manager = TelegramJobQueueManager()
