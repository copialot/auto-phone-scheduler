from datetime import datetime
from typing import Any
from pydantic import BaseModel
from app.models.execution import ExecutionStatus


class ExecutionResponse(BaseModel):
    id: int
    task_id: int
    task_name: str | None = None
    status: ExecutionStatus
    started_at: datetime | None
    finished_at: datetime | None
    error_message: str | None

    class Config:
        from_attributes = True


class ExecutionStep(BaseModel):
    step: int
    action: str | dict[str, Any] | None = None  # 支持字符串或对象
    thinking: str | None = None  # 新增思考内容字段
    description: str | None = None
    screenshot: str | None = None
    timestamp: datetime


class ExecutionDetail(ExecutionResponse):
    steps: list[ExecutionStep] | None
    recording_path: str | None
    device_serial: str | None = None  # 执行任务的设备 serial
