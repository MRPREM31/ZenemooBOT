"""
==============================================================================
Zenemoo AI - Telegram Bot Async REST API Client
==============================================================================
Client service layer connecting the Telegram Bot to the FastAPI backend API.
Handles HTTP request retries, multipart image uploads, and backend status queries.
"""

import httpx
import asyncio
from typing import Dict, Any, Optional
from core.logging import logger
from clients.telegram.config import bot_settings
from shared.exceptions.telegram_exception import TelegramBackendCommunicationException


class BackendAPIClient:
    """Async HTTP Client for Zenemoo AI Backend REST APIs using persistent connection pooling."""

    def __init__(self, base_url: str = bot_settings.BACKEND_API_URL):
        self.base_url = base_url.rstrip("/")
        self.timeout = bot_settings.API_TIMEOUT_SECONDS
        self._client: Optional[httpx.AsyncClient] = None

    def get_client(self) -> httpx.AsyncClient:
        """Returns or initializes shared persistent AsyncClient session."""
        if self._client is None or self._client.is_closed:
            timeout_cfg = httpx.Timeout(self.timeout, connect=60.0)
            self._client = httpx.AsyncClient(timeout=timeout_cfg)
        return self._client


    async def close(self) -> None:
        """Closes the underlying HTTP client session gracefully."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def _request(
        self,
        method: str,
        endpoint: str,
        files: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Internal helper sending async HTTP requests with automatic retries over shared client."""
        url = f"{self.base_url}{endpoint}"
        retries = bot_settings.MAX_RETRIES
        client = self.get_client()

        for attempt in range(1, retries + 1):
            try:
                response = await client.request(
                    method=method,
                    url=url,
                    files=files,
                    data=data,
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.warning(
                        f"Backend API [{url}] returned status {response.status_code}: {response.text}"
                    )
                    if attempt == retries:
                        raise TelegramBackendCommunicationException(
                            f"Backend API error ({response.status_code}): {response.text}"
                        )

            except (httpx.RequestError, httpx.TimeoutException) as e:
                logger.error(f"HTTP request attempt {attempt}/{retries} failed for {url}: {e}")
                if attempt == retries:
                    raise TelegramBackendCommunicationException(
                        f"Failed connecting to Zenemoo AI Backend API at '{url}': {e}"
                    )
                await asyncio.sleep(1.0 * attempt)

        raise TelegramBackendCommunicationException("API request retries exhausted.")

    async def check_health(self) -> Dict[str, Any]:
        """Queries /health endpoint on backend API."""
        return await self._request("GET", "/health")

    async def send_image_job(
        self,
        endpoint: str,
        image_bytes: bytes,
        filename: str = "input.jpg",
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Sends image processing job request to backend API endpoint.
        Example endpoints: '/enhance', '/restore', '/remove-bg', '/upscale', '/compress', '/colorize'
        """
        files = {"file": (filename, image_bytes, "image/jpeg")}
        return await self._request("POST", endpoint, files=files, data=params)


# Singleton instance
bot_api_client = BackendAPIClient()
