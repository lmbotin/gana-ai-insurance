"""
Configuration management using pydantic-settings.

Loads settings from environment variables and .env files.
"""

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # OpenAI Configuration
    openai_api_key: str = Field(
        ...,
        description="OpenAI API key for Realtime API and GPT-4o",
    )
    openai_realtime_model: str = Field(
        default="gpt-4o-realtime-preview",
        description="Model to use for OpenAI Realtime API",
    )
    openai_extraction_model: str = Field(
        default="gpt-4o",
        description="Model to use for FNOL extraction",
    )
    openai_realtime_voice: str = Field(
        default="shimmer",
        description="Voice to use for OpenAI Realtime (alloy, echo, fable, onyx, nova, shimmer). Shimmer is warm and friendly.",
    )
    
    # Twilio Configuration
    twilio_account_sid: Optional[str] = Field(
        default=None,
        description="Twilio Account SID",
    )
    twilio_auth_token: Optional[str] = Field(
        default=None,
        description="Twilio Auth Token",
    )
    
    # Public URLs (for Twilio webhooks)
    public_base_url: str = Field(
        default="http://localhost:8000",
        description="Public HTTP URL for Twilio webhooks",
    )
    public_wss_base_url: str = Field(
        default="ws://localhost:8000",
        description="Public WebSocket URL for Twilio Media Streams",
    )
    
    # Server Configuration
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    debug: bool = Field(default=False, description="Enable debug mode")

    # CORS Configuration (for web chat frontend)
    cors_allowed_origins: str = Field(
        default="*",
        description="Comma-separated list of allowed CORS origins. Use '*' for all (dev only).",
    )

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins into a list."""
        if self.cors_allowed_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_allowed_origins.split(",") if origin.strip()]
    
    # Voice Agent Configuration
    silence_duration_ms: int = Field(
        default=600,
        description="Silence duration before VAD triggers end of speech (ms). 600ms is balanced.",
    )
    # Note: welcome_message is no longer used - the AI agent handles the greeting naturally
    welcome_message: str = Field(
        default="",
        description="Deprecated - AI agent now handles greeting directly",
    )
    
    @property
    def openai_realtime_url(self) -> str:
        """Get the OpenAI Realtime WebSocket URL."""
        return f"wss://api.openai.com/v1/realtime?model={self.openai_realtime_model}"
    
    @property
    def twilio_stream_url(self) -> str:
        """Get the Twilio Media Stream WebSocket URL."""
        base = self.public_wss_base_url.rstrip("/")
        return f"{base}/twilio/stream"


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Uses LRU cache to avoid re-reading environment on every call.
    """
    return Settings()


# Convenience access
settings = get_settings()
