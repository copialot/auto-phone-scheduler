from datetime import datetime
from sqlalchemy import String, Boolean, Integer, Text, DateTime
from sqlalchemy.orm import Mapped, mapped_column
from app.database import Base


class DeviceConfig(Base):
    """设备配置模型，存储设备的唤醒和解锁配置"""
    __tablename__ = "device_configs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    device_serial: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)

    # 唤醒配置
    wake_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    # 唤醒命令类型: keyevent (默认使用 KEYCODE_WAKEUP)
    wake_command: Mapped[str | None] = mapped_column(Text, nullable=True)

    # 解锁配置
    unlock_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    # 解锁类型: swipe (滑动) 或 longpress (长按)
    unlock_type: Mapped[str | None] = mapped_column(String(20), nullable=True)
    # 滑动/长按起始坐标
    unlock_start_x: Mapped[int | None] = mapped_column(Integer, nullable=True)
    unlock_start_y: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # 滑动结束坐标 (长按时与起始坐标相同)
    unlock_end_x: Mapped[int | None] = mapped_column(Integer, nullable=True)
    unlock_end_y: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # 滑动/长按时长 (毫秒)
    unlock_duration: Mapped[int] = mapped_column(Integer, default=300)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
