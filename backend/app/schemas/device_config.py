from datetime import datetime
from pydantic import BaseModel, Field
from typing import Literal


class DeviceConfigBase(BaseModel):
    device_serial: str = Field(..., max_length=100)
    wake_enabled: bool = True
    wake_command: str | None = None
    screen_guard_enabled: bool = False
    unlock_enabled: bool = False
    unlock_type: Literal["swipe", "longpress"] | None = None
    unlock_start_x: int | None = None
    unlock_start_y: int | None = None
    unlock_end_x: int | None = None
    unlock_end_y: int | None = None
    unlock_duration: int = 300


class DeviceConfigCreate(DeviceConfigBase):
    pass


class DeviceConfigUpdate(BaseModel):
    wake_enabled: bool | None = None
    wake_command: str | None = None
    screen_guard_enabled: bool | None = None
    unlock_enabled: bool | None = None
    unlock_type: Literal["swipe", "longpress"] | None = None
    unlock_start_x: int | None = None
    unlock_start_y: int | None = None
    unlock_end_x: int | None = None
    unlock_end_y: int | None = None
    unlock_duration: int | None = None


class DeviceConfigResponse(DeviceConfigBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
