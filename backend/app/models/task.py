from datetime import datetime
from sqlalchemy import String, Boolean, Text, DateTime, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base


class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    command: Mapped[str] = mapped_column(Text, nullable=False)
    cron_expression: Mapped[str] = mapped_column(String(100), nullable=False)
    # 任务执行时区，IANA 时区名称如 "Asia/Shanghai"，为空则使用服务器本地时区
    timezone: Mapped[str | None] = mapped_column(String(50), nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    notify_on_success: Mapped[bool] = mapped_column(Boolean, default=False)
    notify_on_failure: Mapped[bool] = mapped_column(Boolean, default=True)
    # 通知渠道 ID 列表，空列表表示使用所有启用的渠道
    notification_channel_ids: Mapped[list[int] | None] = mapped_column(JSON, nullable=True)
    # 敏感操作自动确认（定时任务默认自动确认，调试模式需要手动确认）
    auto_confirm_sensitive: Mapped[bool] = mapped_column(Boolean, default=True)
    # 指定执行设备 serial，为空则使用全局设置的设备
    device_serial: Mapped[str | None] = mapped_column(String(100), nullable=True)
    # 随机执行时间区间（分钟），在 cron 触发后延迟 0~random_delay_minutes 分钟执行
    random_delay_minutes: Mapped[int | None] = mapped_column(nullable=True, default=None)
    # 执行前唤醒设备（使用设备配置中的唤醒命令）
    wake_before_run: Mapped[bool] = mapped_column(Boolean, default=True)
    # 执行前解锁设备（使用设备配置中的解锁命令）
    unlock_before_run: Mapped[bool] = mapped_column(Boolean, default=True)
    # 执行完成后返回主屏幕
    go_home_after_run: Mapped[bool] = mapped_column(Boolean, default=False)
    # 任务专属最大步数，为空则使用全局设置
    max_steps: Mapped[int | None] = mapped_column(nullable=True, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    executions: Mapped[list["Execution"]] = relationship(
        "Execution", back_populates="task", cascade="all, delete-orphan"
    )
