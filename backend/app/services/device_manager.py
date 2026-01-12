"""设备连接管理服务 - WiFi 设备保活和自动重连"""
import asyncio
import logging
import secrets
import socket
import string
import threading
from datetime import datetime
from typing import Callable

from app.services.adb import run_adb

logger = logging.getLogger(__name__)


def generate_pairing_credentials() -> tuple[str, str]:
    """生成配对凭证（服务名和密码）

    Returns:
        (service_name, password)
    """
    # 生成随机服务名（类似 Android Studio 的格式）
    service_name = "studio-" + "".join(secrets.choice(string.ascii_lowercase + string.digits) for _ in range(8))
    # 生成 6 位数字密码
    password = "".join(secrets.choice(string.digits) for _ in range(6))
    return service_name, password


def generate_qr_code_content(service_name: str, password: str) -> str:
    """生成 ADB 无线调试配对的二维码内容

    格式: WIFI:T:ADB;S:{service_name};P:{password};;

    Args:
        service_name: 服务名称
        password: 配对密码

    Returns:
        二维码内容字符串
    """
    return f"WIFI:T:ADB;S:{service_name};P:{password};;"


def get_local_ip() -> str:
    """获取本机局域网 IP 地址"""
    try:
        # 创建一个 UDP socket 来获取本地 IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


class QRCodePairingSession:
    """二维码配对会话

    工作流程：
    1. 生成配对凭证（服务名和密码）
    2. 注册 mDNS 服务 (_adb-tls-pairing._tcp.local.)
    3. 监听来自 Android 设备的配对请求
    4. 执行 adb pair 完成配对
    """

    def __init__(self, service_name: str, password: str, timeout: int = 120):
        self.service_name = service_name
        self.password = password
        self.timeout = timeout
        self._zeroconf = None
        self._service_info = None
        self._browser = None
        self._paired_device: dict | None = None
        self._stop_event = threading.Event()
        self._listener = None

    def start(self) -> str:
        """启动配对会话，返回二维码内容"""
        try:
            from zeroconf import ServiceInfo, Zeroconf, ServiceBrowser

            # 获取本机 IP
            local_ip = get_local_ip()
            logger.info(f"本机 IP: {local_ip}")

            # 初始化 Zeroconf
            self._zeroconf = Zeroconf()

            # 获取当前事件循环（用于跨线程调度）
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            # 创建监听器来发现配对服务
            self._listener = PairingServiceListener(
                self.service_name,
                self.password,
                self._on_device_paired,
                loop=loop  # 传递事件循环
            )

            # 监听 _adb-tls-pairing._tcp.local. 服务
            self._browser = ServiceBrowser(
                self._zeroconf,
                "_adb-tls-pairing._tcp.local.",
                self._listener
            )

            logger.info(f"开始监听 ADB 配对服务，服务名: {self.service_name}")

            return generate_qr_code_content(self.service_name, self.password)

        except ImportError:
            logger.error("zeroconf 库未安装，请安装: pip install zeroconf")
            raise RuntimeError("zeroconf 库未安装")
        except Exception as e:
            logger.error(f"启动配对会话失败: {e}")
            raise

    def _on_device_paired(self, host: str, port: int):
        """设备配对回调"""
        self._paired_device = {"host": host, "port": port}
        self._stop_event.set()

    def wait_for_pairing(self, timeout: int | None = None) -> dict | None:
        """等待配对完成

        Returns:
            配对成功返回 {"host": str, "port": int}，超时返回 None
        """
        timeout = timeout or self.timeout
        if self._stop_event.wait(timeout):
            return self._paired_device
        return None

    def stop(self):
        """停止配对会话"""
        self._stop_event.set()
        if self._browser:
            # ServiceBrowser 没有 cancel 方法，通过关闭 zeroconf 来停止
            pass
        if self._zeroconf:
            self._zeroconf.close()
            self._zeroconf = None
        logger.info("配对会话已停止")

    @property
    def paired_device(self) -> dict | None:
        return self._paired_device


class PairingServiceListener:
    """mDNS 服务监听器，用于发现 Android 设备的配对请求"""

    def __init__(self, expected_service_name: str, password: str,
                 on_paired_callback: Callable[[str, int], None],
                 loop: asyncio.AbstractEventLoop | None = None):
        self.expected_service_name = expected_service_name
        self.password = password
        self.on_paired_callback = on_paired_callback
        self._paired = False
        self._loop = loop  # 主线程的事件循环

    def add_service(self, zc, type_: str, name: str):
        """发现新服务时调用（在 zeroconf 线程中执行）"""
        if self._paired:
            return

        logger.info(f"发现 mDNS 服务: {name}")

        # 检查服务名是否匹配（Android 设备会发布以我们的服务名为前缀的服务）
        if self.expected_service_name in name:
            try:
                info = zc.get_service_info(type_, name)
                if info:
                    # 获取 IP 和端口
                    addresses = info.parsed_addresses()
                    if addresses:
                        host = addresses[0]
                        port = info.port

                        logger.info(f"找到匹配的配对服务: {host}:{port}")

                        # 在主线程事件循环中执行异步配对
                        if self._loop and self._loop.is_running():
                            asyncio.run_coroutine_threadsafe(
                                self._do_pair(host, port),
                                self._loop
                            )
                        else:
                            # 如果没有事件循环，直接同步执行
                            threading.Thread(
                                target=self._do_pair_sync,
                                args=(host, port),
                                daemon=True
                            ).start()
            except Exception as e:
                logger.error(f"解析服务信息失败: {e}")

    def _do_pair_sync(self, host: str, port: int):
        """同步执行配对（在新线程中）"""
        if self._paired:
            return
        # 创建新的事件循环执行配对
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            success, message = loop.run_until_complete(pair_device(host, port, self.password))
            if success:
                logger.info(f"配对成功: {message}")
                self._paired = True
                self.on_paired_callback(host, port)
            else:
                logger.warning(f"配对失败: {message}")
        finally:
            loop.close()

    async def _do_pair(self, host: str, port: int):
        """执行配对"""
        if self._paired:
            return

        success, message = await pair_device(host, port, self.password)
        if success:
            logger.info(f"配对成功: {message}")
            self._paired = True
            self.on_paired_callback(host, port)
        else:
            logger.warning(f"配对失败: {message}")

    def remove_service(self, zc, type_: str, name: str):
        """服务移除时调用"""
        pass

    def update_service(self, zc, type_: str, name: str):
        """服务更新时调用"""
        pass


class DeviceConnectionManager:
    """WiFi 设备连接管理器

    功能:
    1. 定期检查 WiFi 设备连接状态
    2. 自动重连断开的 WiFi 设备
    3. 维护已知 WiFi 设备列表
    """

    _instance: "DeviceConnectionManager | None" = None

    # 检查间隔（秒）
    CHECK_INTERVAL = 30
    # 重连间隔（秒）
    RECONNECT_INTERVAL = 10
    # 最大重连尝试次数（达到后会重置计数器继续尝试，但间隔更长）
    MAX_RECONNECT_ATTEMPTS = 5
    # 失败后的重连间隔（秒）- 达到上限后使用此间隔
    FAILED_RECONNECT_INTERVAL = 120

    def __init__(self):
        # 已知的 WiFi 设备地址 -> 设备信息
        self._wifi_devices: dict[str, dict] = {}
        # 正在重连的设备
        self._reconnecting: set[str] = set()
        # 后台任务
        self._monitor_task: asyncio.Task | None = None
        # 运行标志
        self._running = False
        # 状态变化回调
        self._on_status_change: Callable[[str, str, str], None] | None = None

    @classmethod
    def get_instance(cls) -> "DeviceConnectionManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def set_status_callback(self, callback: Callable[[str, str, str], None]):
        """设置状态变化回调

        callback(address, old_status, new_status)
        """
        self._on_status_change = callback

    async def start(self):
        """启动连接管理器"""
        if self._running:
            return

        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("设备连接管理器已启动")

    async def stop(self):
        """停止连接管理器"""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
        logger.info("设备连接管理器已停止")

    def register_device(self, address: str, model: str | None = None):
        """注册一个 WiFi 设备（用于保活和重连）

        Args:
            address: 设备地址 (host:port)
            model: 设备型号（可选）
        """
        if address not in self._wifi_devices:
            self._wifi_devices[address] = {
                "address": address,
                "model": model,
                "status": "connected",
                "last_seen": datetime.now(),
                "reconnect_attempts": 0,
            }
            logger.info(f"已注册 WiFi 设备: {address}")

    def unregister_device(self, address: str):
        """取消注册设备（不再进行保活和重连）"""
        if address in self._wifi_devices:
            del self._wifi_devices[address]
            logger.info(f"已取消注册 WiFi 设备: {address}")

    def get_registered_devices(self) -> list[dict]:
        """获取所有已注册的设备"""
        return list(self._wifi_devices.values())

    async def _monitor_loop(self):
        """后台监控循环"""
        while self._running:
            try:
                await self._check_connections()
            except Exception as e:
                logger.error(f"检查设备连接时出错: {e}")

            await asyncio.sleep(self.CHECK_INTERVAL)

    async def _check_connections(self):
        """检查所有已注册设备的连接状态"""
        if not self._wifi_devices:
            return

        # 获取当前已连接的设备
        connected_devices = await self._get_connected_wifi_devices()
        connected_addresses = {d["serial"] for d in connected_devices}

        # 检查每个已注册设备
        for address, device_info in list(self._wifi_devices.items()):
            old_status = device_info.get("status", "unknown")

            if address in connected_addresses:
                # 设备在线
                device_info["status"] = "connected"
                device_info["last_seen"] = datetime.now()
                device_info["reconnect_attempts"] = 0

                if old_status != "connected":
                    logger.info(f"设备 {address} 已恢复连接")
                    if self._on_status_change:
                        self._on_status_change(address, old_status, "connected")
            else:
                # 设备离线，尝试重连
                if address not in self._reconnecting:
                    device_info["status"] = "disconnected"
                    if old_status == "connected":
                        logger.warning(f"设备 {address} 已断开连接，将尝试重连")
                        if self._on_status_change:
                            self._on_status_change(address, old_status, "disconnected")

                    # 触发重连
                    asyncio.create_task(self._reconnect_device(address))

    async def _get_connected_wifi_devices(self) -> list[dict]:
        """获取当前已连接的 WiFi 设备

        注意：Android 15+ 无线调试的设备名可能包含空格，如：
        adb-2233fd19-9mU6UL (2)._adb-tls-connect._tcp device

        需要使用正则表达式正确解析。
        """
        import re

        try:
            stdout, _ = await run_adb("devices", "-l")
            output = stdout.decode()

            devices = []
            lines = output.strip().split("\n")[1:]

            # 匹配设备行的正则：serial + 空白 + status + 可选的其他信息
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

                    # 只处理 WiFi 设备（格式为 host:port）
                    if ":" in serial and not serial.startswith("emulator"):
                        if status == "device":
                            devices.append({"serial": serial, "status": status})

            return devices
        except Exception as e:
            logger.error(f"获取设备列表失败: {e}")
            return []

    def _is_remote_adb_server(self) -> bool:
        """检查是否使用远程 ADB server"""
        from app.services.adb import _parse_adb_socket
        host, port = _parse_adb_socket()
        return host is not None and port is not None

    async def _reconnect_device(self, address: str):
        """尝试重连设备"""
        if address in self._reconnecting:
            return

        if address not in self._wifi_devices:
            return

        device_info = self._wifi_devices[address]

        # 检查重连尝试次数 - 达到上限后重置计数器继续尝试，但等待更长时间
        if device_info["reconnect_attempts"] >= self.MAX_RECONNECT_ATTEMPTS:
            logger.info(f"设备 {address} 重连尝试次数已达上限，等待 {self.FAILED_RECONNECT_INTERVAL}s 后重置计数器继续尝试")
            device_info["status"] = "waiting"
            device_info["reconnect_attempts"] = 0
            if self._on_status_change:
                self._on_status_change(address, "disconnected", "waiting")
            # 等待较长时间后继续
            await asyncio.sleep(self.FAILED_RECONNECT_INTERVAL)

        self._reconnecting.add(address)
        device_info["status"] = "reconnecting"
        device_info["reconnect_attempts"] += 1

        try:
            attempt = device_info["reconnect_attempts"]
            logger.info(f"正在重连设备 {address} (尝试 {attempt}/{self.MAX_RECONNECT_ATTEMPTS})")

            # 先断开连接
            await run_adb("disconnect", address)
            await asyncio.sleep(1)

            # 第一次尝试直接连接
            stdout, stderr = await run_adb("connect", address)
            output = stdout.decode() + stderr.decode()

            if "connected" in output.lower() and "cannot" not in output.lower():
                logger.info(f"设备 {address} 重连成功")
                device_info["status"] = "connected"
                device_info["last_seen"] = datetime.now()
                device_info["reconnect_attempts"] = 0
                if self._on_status_change:
                    self._on_status_change(address, "reconnecting", "connected")
            else:
                # 连接失败，只在本地 ADB server 模式下尝试重置
                if not self._is_remote_adb_server():
                    logger.warning(f"设备 {address} 连接失败: {output.strip()}，尝试重置本地 ADB server")
                    await run_adb("kill-server")
                    await asyncio.sleep(2)
                    await run_adb("start-server")
                    await asyncio.sleep(1)

                    # 再次尝试连接
                    stdout, stderr = await run_adb("connect", address)
                    output = stdout.decode() + stderr.decode()

                    if "connected" in output.lower() and "cannot" not in output.lower():
                        logger.info(f"设备 {address} 重连成功（通过重置 ADB server）")
                        device_info["status"] = "connected"
                        device_info["last_seen"] = datetime.now()
                        device_info["reconnect_attempts"] = 0
                        if self._on_status_change:
                            self._on_status_change(address, "reconnecting", "connected")
                        return
                else:
                    logger.warning(f"设备 {address} 连接失败: {output.strip()}（远程 ADB server 模式，跳过重置）")

                # 标记为断开，等待下次检查时重试
                device_info["status"] = "disconnected"

                # 延迟后再尝试
                if device_info["reconnect_attempts"] < self.MAX_RECONNECT_ATTEMPTS:
                    await asyncio.sleep(self.RECONNECT_INTERVAL)
        except Exception as e:
            logger.error(f"重连设备 {address} 时出错: {e}")
            device_info["status"] = "disconnected"
        finally:
            self._reconnecting.discard(address)

    async def force_reconnect(self, address: str) -> tuple[bool, str]:
        """强制重连设备（手动触发）"""
        if address in self._wifi_devices:
            # 重置重连计数器
            self._wifi_devices[address]["reconnect_attempts"] = 0
            self._wifi_devices[address]["status"] = "disconnected"
        else:
            # 如果不在列表中，先注册
            self.register_device(address)

        # 执行重连
        await self._reconnect_device(address)

        # 返回结果
        device_info = self._wifi_devices.get(address, {})
        if device_info.get("status") == "connected":
            return True, f"设备 {address} 重连成功"
        else:
            return False, f"设备 {address} 重连失败"

    def reset_reconnect_counter(self, address: str):
        """重置设备的重连计数器"""
        if address in self._wifi_devices:
            self._wifi_devices[address]["reconnect_attempts"] = 0
            if self._wifi_devices[address].get("status") == "failed":
                self._wifi_devices[address]["status"] = "disconnected"


async def pair_device(host: str, port: int, pairing_code: str) -> tuple[bool, str]:
    """通过配对码配对设备（Android 11+ 无线调试）

    Args:
        host: 设备 IP 地址
        port: 配对端口（不是连接端口）
        pairing_code: 6位配对码

    Returns:
        (success, message)
    """
    address = f"{host}:{port}"

    try:
        # 使用 adb pair 命令
        process = await asyncio.create_subprocess_exec(
            "adb", "pair", address,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # 发送配对码
        stdout, stderr = await process.communicate(input=f"{pairing_code}\n".encode())
        output = stdout.decode() + stderr.decode()

        if "successfully paired" in output.lower():
            return True, f"配对成功！设备 {host} 已配对"
        elif "failed" in output.lower():
            return False, f"配对失败: {output.strip()}"
        else:
            return False, f"配对结果未知: {output.strip()}"
    except Exception as e:
        return False, f"配对错误: {str(e)}"
