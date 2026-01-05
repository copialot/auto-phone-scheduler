import logging
import subprocess
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScreenGuardConfig:
    enabled: bool = False
    wake: bool = True
    unlock: bool = True
    wake_command: str | None = None
    unlock_type: str | None = None  # "swipe" | "longpress"
    unlock_start_x: int | None = None
    unlock_start_y: int | None = None
    unlock_end_x: int | None = None
    unlock_end_y: int | None = None
    unlock_duration: int = 300


_thread_local = threading.local()


def set_screen_guard_config(config: ScreenGuardConfig | None) -> None:
    _thread_local.screen_guard_config = config


def get_screen_guard_config() -> ScreenGuardConfig | None:
    return getattr(_thread_local, "screen_guard_config", None)


def _adb_cmd_prefix(device_id: str | None) -> list[str]:
    cmd = ["adb"]
    if device_id:
        cmd.extend(["-s", device_id])
    return cmd


def _run_adb_shell(
    device_id: str | None,
    *args: str,
    timeout: float = 10.0,
) -> str:
    cmd = _adb_cmd_prefix(device_id)
    cmd.extend(["shell", *args])
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return (proc.stdout or "") + (proc.stderr or "")


def _is_screen_locked(device_id: str | None) -> tuple[bool, bool]:
    """
    检测屏幕是否锁定
    返回: (屏幕是否亮着, 是否在锁屏界面)
    """
    try:
        screen_on = False
        is_locked = False

        # 方法1: window policy（较准确）
        try:
            policy_output = _run_adb_shell(device_id, "dumpsys", "window", "policy")
            if "mScreenOnFully=true" in policy_output:
                screen_on = True
            if "mInputRestricted=true" in policy_output:
                is_locked = True
        except Exception:
            pass

        # 方法2: power 状态（补充屏幕亮灭判断）
        if not screen_on:
            try:
                power_output = _run_adb_shell(device_id, "dumpsys", "power")
                if any(
                    x in power_output
                    for x in (
                        "mWakefulness=Awake",
                        "Display Power: state=ON",
                        "mHoldingDisplaySuspendBlocker=true",
                    )
                ):
                    screen_on = True
                if "mWakefulness=Asleep" in power_output or "mWakefulness=Dozing" in power_output:
                    screen_on = False
            except Exception:
                pass

        # 方法3: window 状态（补充锁屏判断）
        if not is_locked:
            try:
                window_output = _run_adb_shell(device_id, "dumpsys", "window")
                if any(
                    x in window_output
                    for x in (
                        "mShowingLockscreen=true",
                        "mDreamingLockscreen=true",
                        "isStatusBarKeyguard=true",
                    )
                ):
                    is_locked = True
            except Exception:
                pass

        return screen_on, is_locked
    except Exception:
        # 检测失败：保守认为需要唤醒+解锁
        return False, True


def ensure_device_awake(device_id: str | None) -> None:
    """
    屏幕守护：在截图/动作执行前调用，必要时唤醒与解锁。

    - 仅当当前线程通过 set_screen_guard_config() 开启 enabled 后生效
    - 失败会抛异常，让上层任务尽快以明确原因失败（避免“黑屏乱点”）
    """
    config = get_screen_guard_config()
    if not config or not config.enabled:
        return

    screen_on, is_locked = _is_screen_locked(device_id)

    # 唤醒（仅在息屏时）
    if config.wake and not screen_on:
        logger.info("[ScreenGuard] Screen off detected, waking device...")
        if config.wake_command:
            # 用 sh -c 支持带空格的自定义命令
            _run_adb_shell(device_id, "sh", "-c", config.wake_command)
        else:
            _run_adb_shell(device_id, "input", "keyevent", "KEYCODE_WAKEUP")
        time.sleep(0.3)
        screen_on, is_locked = _is_screen_locked(device_id)

    # 解锁（仅在锁屏时）
    if config.unlock and is_locked:
        if not config.unlock_type:
            raise RuntimeError("屏幕守护已开启，但未配置解锁方式（swipe/longpress）")
        if config.unlock_start_x is None or config.unlock_start_y is None:
            raise RuntimeError("屏幕守护已开启，但未配置解锁起点坐标")

        start_x = config.unlock_start_x
        start_y = config.unlock_start_y
        duration = int(config.unlock_duration or 300)

        if config.unlock_type == "swipe":
            end_x = config.unlock_end_x if config.unlock_end_x is not None else start_x
            end_y = config.unlock_end_y if config.unlock_end_y is not None else start_y
        elif config.unlock_type == "longpress":
            end_x = start_x
            end_y = start_y
        else:
            raise RuntimeError(f"不支持的解锁方式: {config.unlock_type}")

        max_retries = 3
        for attempt in range(max_retries):
            logger.info("[ScreenGuard] Device locked, unlocking... (%d/%d)", attempt + 1, max_retries)
            _run_adb_shell(
                device_id,
                "input",
                "swipe",
                str(start_x),
                str(start_y),
                str(end_x),
                str(end_y),
                str(duration),
            )
            time.sleep(0.5)
            _, still_locked = _is_screen_locked(device_id)
            if not still_locked:
                return
            if attempt < max_retries - 1:
                time.sleep(0.5)

        raise RuntimeError(f"设备解锁失败：重试 {max_retries} 次后仍处于锁屏状态")

