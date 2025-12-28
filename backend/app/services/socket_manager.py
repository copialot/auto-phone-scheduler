"""Socket.IO 管理器 - 用于实时视频流"""

import asyncio
import logging
import re
import sys

import socketio

from app.services.adb import run_adb
from app.services.scrcpy_stream import ScrcpyStreamer, set_streamer, remove_streamer


async def get_device_resolution(device_id: str) -> tuple[int, int] | None:
    """获取设备的原始分辨率"""
    try:
        stdout, _ = await run_adb("shell", "wm", "size", serial=device_id)
        output = stdout.decode()
        # 输出格式: "Physical size: 1080x2400" 或 "Override size: ..."
        # 优先使用 Physical size
        match = re.search(r'Physical size:\s*(\d+)x(\d+)', output)
        if not match:
            # 如果没有 Physical size，尝试匹配任意 size
            match = re.search(r'(\d+)x(\d+)', output)
        if match:
            return int(match.group(1)), int(match.group(2))
    except Exception as e:
        logging.warning(f"获取设备分辨率失败: {e}")
    return None

# 配置日志输出
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 创建 Socket.IO 服务器
# cors_allowed_origins="*" 允许所有来源（开发环境）
sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins="*",
    cors_credentials=False,  # 使用 * 时不能设置 credentials
    logger=False,
    engineio_logger=False,
)

# 存储每个 session 的流任务
_stream_tasks: dict[str, asyncio.Task] = {}


@sio.event
async def connect(sid, _environ):
    """客户端连接"""
    print(f"[SocketIO] 客户端连接: {sid}", flush=True)
    logger.info(f"客户端连接: {sid}")


@sio.event
async def disconnect(sid):
    """客户端断开"""
    logger.info(f"客户端断开: {sid}")
    # 取消该 session 的流任务
    if sid in _stream_tasks:
        _stream_tasks[sid].cancel()
        del _stream_tasks[sid]


@sio.on("connect-device")
async def connect_device(sid, data):
    """连接设备并开始流式传输"""
    print(f"[SocketIO] connect-device: sid={sid}, data={data}", flush=True)
    logger.info(f"[connect_device] 收到连接请求: sid={sid}, data={data}")

    device_id = data.get("device_id")
    max_size = data.get("maxSize", 1280)
    bit_rate = data.get("bitRate", 4_000_000)

    if not device_id:
        logger.warning(f"[connect_device] 缺少 device_id")
        await sio.emit("error", {"message": "缺少 device_id"}, room=sid)
        return

    logger.info(f"[connect_device] 开始连接设备: {device_id}, sid: {sid}")

    # 取消之前的任务
    if sid in _stream_tasks:
        _stream_tasks[sid].cancel()

    # 为每个连接分配唯一端口（避免冲突）
    port = 27183 + hash(sid) % 1000

    streamer = ScrcpyStreamer(
        device_id=device_id,
        max_size=max_size,
        bit_rate=bit_rate,
        port=port,
    )

    async def stream_video():
        try:
            logger.info(f"[stream_video] 开始启动流: {device_id}")
            await streamer.start()
            logger.info(f"[stream_video] 流启动成功: {device_id}")
            set_streamer(device_id, streamer)

            # 发送元数据
            logger.info(f"[stream_video] 读取视频元数据...")
            metadata = await streamer.read_video_metadata()
            logger.info(f"[stream_video] 元数据: {metadata.width}x{metadata.height}, codec={metadata.codec}")

            # 获取设备原始分辨率（用于触控坐标转换）
            device_resolution = await get_device_resolution(device_id)
            original_width, original_height = device_resolution if device_resolution else (metadata.width, metadata.height)
            logger.info(f"[stream_video] 设备原始分辨率: {original_width}x{original_height}")

            await sio.emit("video-metadata", {
                "deviceName": metadata.device_name,
                "width": metadata.width,
                "height": metadata.height,
                "codec": metadata.codec,
                # 设备原始分辨率，用于触控坐标转换
                "originalWidth": original_width,
                "originalHeight": original_height,
            }, room=sid)

            # 发送视频数据包
            logger.info(f"[stream_video] 开始发送视频数据包...")
            async for packet in streamer.iter_packets():
                await sio.emit("video-data", {
                    "type": packet.type,
                    "data": bytes(packet.data),
                    "keyframe": packet.keyframe,
                    "pts": packet.pts,
                }, room=sid)

        except asyncio.CancelledError:
            logger.info(f"[stream_video] 任务被取消: {sid}")
        except Exception as e:
            logger.exception(f"[stream_video] 流错误: {e}")
            await sio.emit("error", {"message": str(e)}, room=sid)
        finally:
            logger.info(f"[stream_video] 清理资源: {device_id}")
            streamer.stop()
            remove_streamer(device_id)

    logger.info(f"[connect_device] 创建流任务...")
    task = asyncio.create_task(stream_video())
    _stream_tasks[sid] = task
    logger.info(f"[connect_device] 任务已创建: {sid}")


@sio.on("disconnect-device")
async def disconnect_device(sid, _data):
    """断开设备连接"""
    if sid in _stream_tasks:
        _stream_tasks[sid].cancel()
        del _stream_tasks[sid]
        logger.info(f"设备断开: {sid}")


def create_socket_app(app):
    """创建 Socket.IO ASGI 应用"""
    return socketio.ASGIApp(sio, app)

