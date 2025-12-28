import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from app.config import get_settings
from app.services.adb import get_adb_command, run_adb, run_adb_exec

settings = get_settings()
logger = logging.getLogger(__name__)


class RecorderService:
    """手机屏幕录制服务"""

    def __init__(self, device_serial: str | None = None):
        self.recordings_dir = Path(settings.recordings_dir).resolve()
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        self._recording_process: asyncio.subprocess.Process | None = None
        self._current_file: str | None = None
        self._remote_filename: str | None = None
        self._device_serial: str | None = device_serial

    async def start_recording(self, execution_id: int) -> str:
        """
        开始录制屏幕

        Args:
            execution_id: 执行记录ID

        Returns:
            录制文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"exec_{execution_id}_{timestamp}.mp4"
        filepath = self.recordings_dir / filename
        self._current_file = str(filepath)
        self._remote_filename = filename

        logger.info(f"[Recorder] 开始录制: {self._current_file}")

        # 使用 adb screenrecord 录制
        # 注意：screenrecord 最长录制 3 分钟，需要循环录制
        # 优化参数：--bit-rate 降低码率到 2Mbps（默认约 20Mbps），大幅减小文件大小
        try:
            self._recording_process = await run_adb_exec(
                "shell", "screenrecord",
                "--bit-rate", "2000000",  # 2 Mbps
                "--time-limit", "180",  # 3分钟
                f"/sdcard/{filename}",
                serial=self._device_serial,
            )
            logger.info(f"[Recorder] screenrecord 进程已启动: PID={self._recording_process.pid}")
        except Exception as e:
            logger.error(f"[Recorder] 启动 screenrecord 失败: {e}")
            self._current_file = None
            self._remote_filename = None
            raise

        return self._current_file

    async def stop_recording(self) -> str | None:
        """
        停止录制并拉取文件

        Returns:
            本地录制文件路径
        """
        if not self._recording_process or not self._current_file or not self._remote_filename:
            logger.warning("[Recorder] 没有正在进行的录制")
            return None

        logger.info(f"[Recorder] 停止录制: {self._remote_filename}")

        # 发送中断信号停止录制
        try:
            self._recording_process.terminate()
            await asyncio.wait_for(self._recording_process.wait(), timeout=5.0)
            logger.info("[Recorder] screenrecord 进程已终止")
        except asyncio.TimeoutError:
            logger.warning("[Recorder] 等待进程终止超时，强制杀死")
            self._recording_process.kill()
            await self._recording_process.wait()
        except ProcessLookupError:
            logger.info("[Recorder] 进程已经结束")

        # 等待一小段时间确保文件写入完成
        await asyncio.sleep(1.0)

        # 从设备拉取录制文件
        pull_cmd = get_adb_command()
        if self._device_serial:
            pull_cmd.extend(["-s", self._device_serial])
        pull_cmd.extend(["pull", f"/sdcard/{self._remote_filename}", self._current_file])
        logger.info(f"[Recorder] 拉取文件: {' '.join(pull_cmd)}")

        try:
            stdout, stderr = await run_adb(
                "pull", f"/sdcard/{self._remote_filename}", self._current_file,
                serial=self._device_serial,
            )
            if stderr and b"error" in stderr.lower():
                logger.error(f"[Recorder] adb pull 失败: {stderr.decode()}")
            else:
                logger.info(f"[Recorder] 文件拉取成功: {stdout.decode().strip()}")
        except asyncio.TimeoutError:
            logger.error("[Recorder] adb pull 超时")
        except Exception as e:
            logger.error(f"[Recorder] adb pull 异常: {e}")

        # 清理设备上的文件
        try:
            await run_adb("shell", "rm", "-f", f"/sdcard/{self._remote_filename}", serial=self._device_serial)
            logger.info("[Recorder] 设备上的录制文件已清理")
        except Exception as e:
            logger.warning(f"[Recorder] 清理设备文件失败: {e}")

        # 检查本地文件是否存在
        result = self._current_file
        if Path(result).exists():
            file_size = Path(result).stat().st_size
            logger.info(f"[Recorder] 录制完成，文件大小: {file_size} bytes")
        else:
            logger.warning(f"[Recorder] 录制文件不存在: {result}")
            result = None

        self._recording_process = None
        self._current_file = None
        self._remote_filename = None

        return result

    async def get_recording_path(self, execution_id: int) -> str | None:
        """获取指定执行记录的录屏文件路径"""
        pattern = f"exec_{execution_id}_*.mp4"
        files = list(self.recordings_dir.glob(pattern))
        if files:
            return str(files[0])
        return None

    def delete_recording(self, filepath: str) -> bool:
        """删除录制文件"""
        try:
            os.remove(filepath)
            return True
        except Exception:
            return False
