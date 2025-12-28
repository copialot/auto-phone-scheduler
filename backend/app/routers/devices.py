import asyncio
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
from typing import Literal
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas.device import DeviceInfo
from app.services.adb import run_adb, run_adb_exec
from app.services.streamer import generate_mjpeg_stream
from app.models.execution import Execution, ExecutionStatus

router = APIRouter(prefix="/api/devices", tags=["devices"])


class ConnectRequest(BaseModel):
    address: str  # host:port 格式，例如 192.168.1.100:5555


class ConnectResponse(BaseModel):
    success: bool
    message: str
    serial: str | None = None


class KeyEventRequest(BaseModel):
    key: Literal["home", "back", "app_switch"]  # 按键类型


class SwipeRequest(BaseModel):
    start_x: int
    start_y: int
    end_x: int
    end_y: int
    duration: int = 300  # 滑动时长(ms)


class TapRequest(BaseModel):
    x: int
    y: int


async def get_connected_devices() -> list[DeviceInfo]:
    """获取已连接的ADB设备"""
    stdout, _ = await run_adb("devices", "-l")
    output = stdout.decode()

    devices = []
    lines = output.strip().split("\n")[1:]  # 跳过第一行标题

    for line in lines:
        if not line.strip():
            continue

        parts = line.split()
        if len(parts) >= 2:
            serial = parts[0]
            status = parts[1]

            # 解析额外信息
            model = None
            product = None
            for part in parts[2:]:
                if part.startswith("model:"):
                    model = part.split(":")[1]
                elif part.startswith("product:"):
                    product = part.split(":")[1]

            devices.append(
                DeviceInfo(
                    serial=serial,
                    status=status,
                    model=model,
                    product=product,
                )
            )

    return devices


@router.get("", response_model=list[DeviceInfo])
async def list_devices():
    """获取已连接设备列表"""
    return await get_connected_devices()


@router.post("/refresh", response_model=list[DeviceInfo])
async def refresh_devices():
    """刷新设备列表（不重启ADB服务器，保持WiFi连接）"""
    # 直接返回当前连接的设备列表，不重启 ADB 服务器
    # 因为重启服务器会断开所有 WiFi 连接的设备
    return await get_connected_devices()


@router.get("/{serial}/stream")
async def stream_device(serial: str):
    """获取设备的实时屏幕流（MJPEG）"""
    return StreamingResponse(
        generate_mjpeg_stream(serial),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/{serial}/screenshot")
async def get_screenshot(serial: str):
    """获取设备的单张屏幕截图"""
    stdout, _ = await run_adb("exec-out", "screencap", "-p", serial=serial)

    if stdout and len(stdout) > 100:
        return Response(
            content=stdout,
            media_type="image/png",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )

    return Response(status_code=204)


@router.post("/connect", response_model=ConnectResponse)
async def connect_device(request: ConnectRequest):
    """连接远程设备（WiFi/局域网）

    支持格式：
    - host:port (例如 192.168.1.100:5555)
    - host (默认使用端口 5555)
    """
    address = request.address.strip()

    # 如果没有指定端口，添加默认端口
    if ":" not in address:
        address = f"{address}:5555"

    try:
        stdout, stderr = await run_adb("connect", address)
        output = stdout.decode() + stderr.decode()

        # 检查连接结果
        if "connected" in output.lower():
            # 连接成功，等待设备就绪
            await asyncio.sleep(1)
            return ConnectResponse(
                success=True,
                message=f"成功连接到 {address}",
                serial=address,
            )
        elif "already connected" in output.lower():
            return ConnectResponse(
                success=True,
                message=f"设备 {address} 已经连接",
                serial=address,
            )
        else:
            return ConnectResponse(
                success=False,
                message=f"连接失败: {output.strip()}",
            )
    except Exception as e:
        return ConnectResponse(
            success=False,
            message=f"连接错误: {str(e)}",
        )


@router.post("/disconnect/{serial}", response_model=ConnectResponse)
async def disconnect_device(serial: str):
    """断开远程设备连接

    仅支持断开通过 WiFi/网络连接的设备（host:port 格式）
    """
    # 检查是否是网络设备（包含冒号表示 host:port）
    if ":" not in serial or serial.startswith("emulator"):
        return ConnectResponse(
            success=False,
            message="只能断开网络连接的设备",
        )

    try:
        stdout, stderr = await run_adb("disconnect", serial)
        output = stdout.decode() + stderr.decode()

        if "disconnected" in output.lower() or "error" not in output.lower():
            return ConnectResponse(
                success=True,
                message=f"已断开 {serial}",
                serial=serial,
            )
        else:
            return ConnectResponse(
                success=False,
                message=f"断开失败: {output.strip()}",
            )
    except Exception as e:
        return ConnectResponse(
            success=False,
            message=f"断开错误: {str(e)}",
        )


# Android 按键码映射
KEY_CODES = {
    "home": 3,       # KEYCODE_HOME
    "back": 4,       # KEYCODE_BACK
    "app_switch": 187,  # KEYCODE_APP_SWITCH (最近任务)
}


@router.post("/{serial}/keyevent")
async def send_key_event(serial: str, request: KeyEventRequest):
    """发送按键事件到设备"""
    key_code = KEY_CODES.get(request.key)
    if key_code is None:
        raise HTTPException(status_code=400, detail=f"不支持的按键: {request.key}")

    try:
        await run_adb("shell", "input", "keyevent", str(key_code), serial=serial)
        return {"success": True, "message": f"已发送 {request.key} 按键"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"发送按键失败: {str(e)}")


@router.post("/{serial}/swipe")
async def send_swipe(serial: str, request: SwipeRequest):
    """发送滑动事件到设备"""
    try:
        await run_adb(
            "shell", "input", "swipe",
            str(request.start_x), str(request.start_y),
            str(request.end_x), str(request.end_y),
            str(request.duration),
            serial=serial,
        )
        return {"success": True, "message": "滑动完成"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"滑动失败: {str(e)}")


@router.post("/{serial}/tap")
async def send_tap(serial: str, request: TapRequest):
    """发送点击事件到设备"""
    try:
        await run_adb("shell", "input", "tap", str(request.x), str(request.y), serial=serial)
        return {"success": True, "message": "点击完成"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"点击失败: {str(e)}")


class DeviceBusyStatus(BaseModel):
    is_busy: bool
    execution_id: int | None = None
    task_id: int | None = None
    task_name: str | None = None
    started_at: str | None = None


@router.get("/{serial}/busy-status", response_model=DeviceBusyStatus)
async def get_device_busy_status(serial: str, db: AsyncSession = Depends(get_db)):
    """获取设备的忙碌状态（是否有正在执行的任务）"""
    from sqlalchemy.orm import joinedload

    result = await db.execute(
        select(Execution)
        .options(joinedload(Execution.task))
        .where(
            Execution.device_serial == serial,
            Execution.status == ExecutionStatus.RUNNING,
        )
        .limit(1)
    )
    running_execution = result.scalar_one_or_none()

    if running_execution:
        return DeviceBusyStatus(
            is_busy=True,
            execution_id=running_execution.id,
            task_id=running_execution.task_id,
            task_name=running_execution.task.name if running_execution.task else None,
            started_at=running_execution.started_at.isoformat() if running_execution.started_at else None,
        )

    return DeviceBusyStatus(is_busy=False)


@router.post("/{serial}/release")
async def release_device(serial: str, db: AsyncSession = Depends(get_db)):
    """释放设备（将所有 RUNNING 状态的执行记录标记为失败）

    用于手动清除卡住的任务状态
    """
    from datetime import datetime

    result = await db.execute(
        select(Execution).where(
            Execution.device_serial == serial,
            Execution.status == ExecutionStatus.RUNNING,
        )
    )
    running_executions = result.scalars().all()

    if not running_executions:
        return {"success": True, "message": "设备未被占用", "released_count": 0}

    for execution in running_executions:
        execution.status = ExecutionStatus.FAILED
        execution.finished_at = datetime.utcnow()
        execution.error_message = "手动释放设备"

    await db.commit()

    return {
        "success": True,
        "message": f"已释放 {len(running_executions)} 个执行记录",
        "released_count": len(running_executions),
    }
