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
from app.services.device_manager import (
    DeviceConnectionManager,
    pair_device,
    generate_pairing_credentials,
    generate_qr_code_content,
    QRCodePairingSession,
)
from app.models.execution import Execution, ExecutionStatus

router = APIRouter(prefix="/api/devices", tags=["devices"])


class ConnectRequest(BaseModel):
    address: str  # host:port 格式，例如 192.168.1.100:5555
    register: bool = True  # 是否注册到连接管理器（用于保活和重连）


class ConnectResponse(BaseModel):
    success: bool
    message: str
    serial: str | None = None


class PairRequest(BaseModel):
    host: str  # 设备 IP 地址
    port: int  # 配对端口（不是连接端口）
    pairing_code: str  # 6位配对码


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
    """获取已连接的ADB设备

    注意：Android 15+ 无线调试的设备名可能包含空格，如：
    adb-2233fd19-9mU6UL (2)._adb-tls-connect._tcp device

    需要使用正则表达式正确解析。
    """
    import re

    stdout, _ = await run_adb("devices", "-l")
    output = stdout.decode()

    devices = []
    lines = output.strip().split("\n")[1:]  # 跳过第一行标题

    # 匹配设备行的正则：serial + 空白 + status + 可选的其他信息
    # serial 可能包含空格（如 mDNS 名称），但 status 只能是 device/offline/unauthorized 等
    # 格式示例：
    #   192.168.0.104:40559          device product:... model:... device:...
    #   adb-xxx (2)._adb-tls-connect._tcp device
    device_pattern = re.compile(
        r'^(.+?)\s+(device|offline|unauthorized|recovery|sideload|bootloader|no permissions)(?:\s+(.*))?$'
    )

    for line in lines:
        if not line.strip():
            continue

        match = device_pattern.match(line)
        if match:
            serial = match.group(1).strip()
            status = match.group(2)
            extra_info = match.group(3) or ""

            # 解析额外信息（model:xxx product:xxx 等）
            model = None
            product = None
            for part in extra_info.split():
                if part.startswith("model:"):
                    model = part.split(":", 1)[1]
                elif part.startswith("product:"):
                    product = part.split(":", 1)[1]

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

    连接成功后会自动注册到连接管理器，用于保活和断线重连。
    """
    address = request.address.strip()

    # 如果没有指定端口，添加默认端口
    if ":" not in address:
        address = f"{address}:5555"

    try:
        stdout, stderr = await run_adb("connect", address)
        output = stdout.decode() + stderr.decode()

        # 检查连接结果
        if "connected" in output.lower() and "cannot" not in output.lower():
            # 连接成功，等待设备就绪
            await asyncio.sleep(1)

            # 注册到连接管理器
            if request.register:
                manager = DeviceConnectionManager.get_instance()
                manager.register_device(address)

            return ConnectResponse(
                success=True,
                message=f"成功连接到 {address}",
                serial=address,
            )
        elif "already connected" in output.lower():
            # 已连接也注册一下
            if request.register:
                manager = DeviceConnectionManager.get_instance()
                manager.register_device(address)

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
    断开后会从连接管理器中移除（不再保活和重连）
    """
    # 检查是否是网络设备（包含冒号表示 host:port）
    if ":" not in serial or serial.startswith("emulator"):
        return ConnectResponse(
            success=False,
            message="只能断开网络连接的设备",
        )

    try:
        # 从连接管理器中移除
        manager = DeviceConnectionManager.get_instance()
        manager.unregister_device(serial)

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


# ============ WiFi 设备配对和重连 ============


class QRCodePairingResponse(BaseModel):
    qr_content: str  # 二维码内容
    service_name: str  # 服务名
    password: str  # 配对密码
    session_id: str  # 配对会话 ID


# 存储活动的配对会话
_pairing_sessions: dict[str, QRCodePairingSession] = {}


@router.get("/pair/qrcode", response_model=QRCodePairingResponse)
async def get_pairing_qrcode():
    """生成配对二维码并启动 mDNS 监听（Android 11+ 无线调试）

    工作流程：
    1. 生成配对凭证和二维码
    2. 启动 mDNS 服务监听，等待 Android 设备扫码配对
    3. 返回二维码内容，前端显示二维码供手机扫描

    手机扫码后：
    1. Android 系统会发布 mDNS 配对服务
    2. 服务器监听到服务后自动执行 adb pair 配对
    3. 配对成功后需要调用 /connect 接口连接设备

    使用 /pair/status/{session_id} 轮询配对状态
    """
    import uuid

    service_name, password = generate_pairing_credentials()
    session_id = str(uuid.uuid4())[:8]

    # 创建并启动配对会话
    session = QRCodePairingSession(service_name, password, timeout=120)
    try:
        qr_content = session.start()
        _pairing_sessions[session_id] = session

        # 设置超时自动清理
        async def cleanup_session():
            await asyncio.sleep(130)  # 比配对超时稍长
            if session_id in _pairing_sessions:
                _pairing_sessions[session_id].stop()
                del _pairing_sessions[session_id]

        asyncio.create_task(cleanup_session())

        return QRCodePairingResponse(
            qr_content=qr_content,
            service_name=service_name,
            password=password,
            session_id=session_id,
        )
    except Exception as e:
        session.stop()
        raise HTTPException(status_code=500, detail=str(e))


class PairingStatusResponse(BaseModel):
    status: str  # waiting, paired, timeout, error
    host: str | None = None
    port: int | None = None
    message: str | None = None


@router.get("/pair/status/{session_id}", response_model=PairingStatusResponse)
async def get_pairing_status(session_id: str):
    """获取配对会话状态

    返回值：
    - status: waiting (等待扫码), paired (已配对), timeout (超时), error (错误)
    - host/port: 配对成功时返回设备的 IP 和端口
    """
    session = _pairing_sessions.get(session_id)
    if not session:
        return PairingStatusResponse(
            status="timeout",
            message="配对会话已过期或不存在"
        )

    paired_device = session.paired_device
    if paired_device:
        # 配对成功，清理会话
        session.stop()
        del _pairing_sessions[session_id]
        return PairingStatusResponse(
            status="paired",
            host=paired_device["host"],
            port=paired_device["port"],
            message="配对成功！请使用返回的 IP 和连接端口进行连接"
        )

    return PairingStatusResponse(
        status="waiting",
        message="等待设备扫码配对..."
    )


@router.delete("/pair/{session_id}")
async def cancel_pairing(session_id: str):
    """取消配对会话"""
    session = _pairing_sessions.get(session_id)
    if session:
        session.stop()
        del _pairing_sessions[session_id]
        return {"success": True, "message": "配对已取消"}
    return {"success": False, "message": "配对会话不存在"}


@router.post("/pair", response_model=ConnectResponse)
async def pair_device_endpoint(request: PairRequest):
    """通过配对码配对设备（Android 11+ 无线调试）

    在手机「开发者选项 → 无线调试 → 使用配对码配对设备」中获取：
    - host: 设备 IP 地址
    - port: 配对端口（注意：不是连接端口）
    - pairing_code: 6位配对码

    配对成功后，还需要调用 /connect 接口连接设备（使用无线调试页面显示的连接端口）
    """
    success, message = await pair_device(request.host, request.port, request.pairing_code)

    return ConnectResponse(
        success=success,
        message=message,
        serial=f"{request.host}:{request.port}" if success else None,
    )


class ReconnectResponse(BaseModel):
    success: bool
    message: str


@router.post("/reconnect/{serial}", response_model=ReconnectResponse)
async def reconnect_device(serial: str):
    """强制重连设备

    用于手动触发断线设备的重连
    """
    if ":" not in serial or serial.startswith("emulator"):
        return ReconnectResponse(
            success=False,
            message="只能重连网络设备",
        )

    manager = DeviceConnectionManager.get_instance()
    success, message = await manager.force_reconnect(serial)

    return ReconnectResponse(success=success, message=message)


class RegisteredDevice(BaseModel):
    address: str
    model: str | None = None
    status: str
    last_seen: str | None = None
    reconnect_attempts: int = 0


@router.get("/registered", response_model=list[RegisteredDevice])
async def get_registered_devices():
    """获取所有已注册的 WiFi 设备（用于保活和重连的设备列表）"""
    manager = DeviceConnectionManager.get_instance()
    devices = manager.get_registered_devices()

    return [
        RegisteredDevice(
            address=d["address"],
            model=d.get("model"),
            status=d.get("status", "unknown"),
            last_seen=d["last_seen"].isoformat() if d.get("last_seen") else None,
            reconnect_attempts=d.get("reconnect_attempts", 0),
        )
        for d in devices
    ]


@router.delete("/registered/{serial}")
async def unregister_device(serial: str):
    """取消注册设备（不再保活和重连，但不断开连接）"""
    if ":" not in serial or serial.startswith("emulator"):
        return {"success": False, "message": "只能取消注册网络设备"}

    manager = DeviceConnectionManager.get_instance()
    manager.unregister_device(serial)

    return {"success": True, "message": f"已取消注册 {serial}"}
