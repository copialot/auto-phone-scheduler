from datetime import datetime
from pydantic import BaseModel, Field


class TaskBase(BaseModel):
    name: str = Field(..., max_length=100)
    description: str | None = None
    command: str
    cron_expression: str = Field(..., max_length=100)
    timezone: str | None = None  # IANA 时区名称，如 "Asia/Shanghai"
    enabled: bool = True
    notify_on_success: bool = False
    notify_on_failure: bool = True
    notification_channel_ids: list[int] | None = None  # 空或 None 表示使用所有启用的渠道
    auto_confirm_sensitive: bool = True  # 敏感操作自动确认
    device_serial: str | None = None  # 指定执行设备，为空则使用全局设置
    random_delay_minutes: int | None = None  # 随机延迟时间区间（分钟）
    wake_before_run: bool = True  # 执行前唤醒设备
    unlock_before_run: bool = True  # 执行前解锁设备
    go_home_after_run: bool = False  # 执行完成后返回主屏幕


class TaskCreate(TaskBase):
    pass


class TaskUpdate(BaseModel):
    name: str | None = Field(None, max_length=100)
    description: str | None = None
    command: str | None = None
    cron_expression: str | None = Field(None, max_length=100)
    timezone: str | None = None
    enabled: bool | None = None
    notify_on_success: bool | None = None
    notify_on_failure: bool | None = None
    notification_channel_ids: list[int] | None = None
    auto_confirm_sensitive: bool | None = None
    device_serial: str | None = None
    random_delay_minutes: int | None = None
    wake_before_run: bool | None = None
    unlock_before_run: bool | None = None
    go_home_after_run: bool | None = None


class TaskResponse(TaskBase):
    id: int
    created_at: datetime
    updated_at: datetime
    next_run: str | None = None

    class Config:
        from_attributes = True
