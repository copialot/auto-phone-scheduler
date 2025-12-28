from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.device_config import DeviceConfig
from app.schemas.device_config import (
    DeviceConfigCreate,
    DeviceConfigUpdate,
    DeviceConfigResponse,
)
from app.services.adb import run_adb


class TestResult(BaseModel):
    success: bool
    message: str


class TestWakeRequest(BaseModel):
    wake_command: str | None = None  # 自定义唤醒命令，为空使用默认


class TestUnlockRequest(BaseModel):
    unlock_type: str  # swipe 或 longpress
    unlock_start_x: int
    unlock_start_y: int
    unlock_end_x: int | None = None  # 滑动解锁需要
    unlock_end_y: int | None = None  # 滑动解锁需要
    unlock_duration: int = 300

router = APIRouter(prefix="/api/device-configs", tags=["device-configs"])


@router.get("", response_model=list[DeviceConfigResponse])
async def list_device_configs(db: AsyncSession = Depends(get_db)):
    """获取所有设备配置"""
    result = await db.execute(select(DeviceConfig))
    return result.scalars().all()


@router.get("/{device_serial}", response_model=DeviceConfigResponse | None)
async def get_device_config(device_serial: str, db: AsyncSession = Depends(get_db)):
    """获取指定设备的配置"""
    result = await db.execute(
        select(DeviceConfig).where(DeviceConfig.device_serial == device_serial)
    )
    return result.scalar_one_or_none()


@router.post("", response_model=DeviceConfigResponse)
async def create_device_config(
    config_in: DeviceConfigCreate, db: AsyncSession = Depends(get_db)
):
    """创建设备配置"""
    # 检查是否已存在
    result = await db.execute(
        select(DeviceConfig).where(
            DeviceConfig.device_serial == config_in.device_serial
        )
    )
    existing = result.scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=400, detail="设备配置已存在")

    config = DeviceConfig(**config_in.model_dump())
    db.add(config)
    await db.commit()
    await db.refresh(config)
    return config


@router.put("/{device_serial}", response_model=DeviceConfigResponse)
async def update_device_config(
    device_serial: str,
    config_in: DeviceConfigUpdate,
    db: AsyncSession = Depends(get_db),
):
    """更新设备配置（如果不存在则创建）"""
    result = await db.execute(
        select(DeviceConfig).where(DeviceConfig.device_serial == device_serial)
    )
    config = result.scalar_one_or_none()

    if not config:
        # 创建新配置
        config = DeviceConfig(device_serial=device_serial)
        db.add(config)

    # 更新字段
    update_data = config_in.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(config, field, value)

    await db.commit()
    await db.refresh(config)
    return config


@router.delete("/{device_serial}")
async def delete_device_config(device_serial: str, db: AsyncSession = Depends(get_db)):
    """删除设备配置"""
    result = await db.execute(
        select(DeviceConfig).where(DeviceConfig.device_serial == device_serial)
    )
    config = result.scalar_one_or_none()
    if not config:
        raise HTTPException(status_code=404, detail="设备配置不存在")

    await db.delete(config)
    await db.commit()
    return {"message": "设备配置已删除"}


async def is_screen_locked(device_serial: str) -> tuple[bool, bool]:
    """检测屏幕是否锁定
    返回: (屏幕是否亮着, 是否在锁屏界面)

    使用多种方法检测以提高兼容性：
    1. mScreenOnFully - 屏幕完全亮起状态（最准确）
    2. mWakefulness - 设备唤醒状态
    3. mInputRestricted - 输入受限状态（锁屏时为 true）
    4. mShowingLockscreen / mDreamingLockscreen - 锁屏界面状态
    """
    try:
        screen_on = False
        is_locked = False

        # 方法1: 检查 window policy 状态（最准确）
        try:
            stdout, _ = await run_adb(
                "shell", "dumpsys", "window", "policy",
                serial=device_serial
            )
            policy_output = stdout.decode()
            # mScreenOnFully=true 表示屏幕完全亮起
            if "mScreenOnFully=true" in policy_output:
                screen_on = True
            # mInputRestricted=true 表示输入受限（锁屏状态）
            if "mInputRestricted=true" in policy_output:
                is_locked = True
        except Exception:
            pass

        # 方法2: 如果方法1未检测到屏幕状态，尝试 power 状态
        if not screen_on:
            try:
                stdout, _ = await run_adb(
                    "shell", "dumpsys", "power",
                    serial=device_serial
                )
                power_output = stdout.decode()
                # 检查多种屏幕开启标志
                if any(x in power_output for x in [
                    "mWakefulness=Awake",
                    "Display Power: state=ON",
                    "mHoldingDisplaySuspendBlocker=true",
                ]):
                    screen_on = True
                # 如果 mWakefulness=Asleep 或 Dozing，屏幕肯定关闭
                if "mWakefulness=Asleep" in power_output or "mWakefulness=Dozing" in power_output:
                    screen_on = False
            except Exception:
                pass

        # 方法3: 如果方法1未检测到锁屏状态，尝试 window 状态
        if not is_locked:
            try:
                stdout, _ = await run_adb(
                    "shell", "dumpsys", "window",
                    serial=device_serial
                )
                window_output = stdout.decode()
                # 检查多种锁屏标志
                if any(x in window_output for x in [
                    "mShowingLockscreen=true",
                    "mDreamingLockscreen=true",
                    "isStatusBarKeyguard=true",
                ]):
                    is_locked = True
            except Exception:
                pass

        return screen_on, is_locked
    except Exception:
        # 如果检测失败，假设需要解锁
        return True, True


@router.get("/{device_serial}/screen-status")
async def get_screen_status(device_serial: str):
    """获取设备屏幕状态"""
    screen_on, is_locked = await is_screen_locked(device_serial)
    return {
        "screen_on": screen_on,
        "is_locked": is_locked,
    }


@router.post("/{device_serial}/test-wake", response_model=TestResult)
async def test_wake(device_serial: str, request: TestWakeRequest):
    """测试唤醒设备（使用表单中的配置值）"""
    try:
        if request.wake_command:
            # 使用自定义唤醒命令
            cmd_parts = request.wake_command.split()
            await run_adb("shell", *cmd_parts, serial=device_serial)
        else:
            # 使用默认唤醒命令
            await run_adb("shell", "input", "keyevent", "KEYCODE_WAKEUP", serial=device_serial)

        return TestResult(success=True, message="唤醒命令已发送")
    except Exception as e:
        return TestResult(success=False, message=f"唤醒失败: {str(e)}")


@router.post("/{device_serial}/test-unlock", response_model=TestResult)
async def test_unlock(device_serial: str, request: TestUnlockRequest):
    """测试解锁设备（使用表单中的配置值）"""
    try:
        start_x = request.unlock_start_x
        start_y = request.unlock_start_y
        duration = request.unlock_duration

        if request.unlock_type == "swipe":
            # 滑动解锁
            if request.unlock_end_x is None or request.unlock_end_y is None:
                return TestResult(success=False, message="滑动解锁需要终点坐标")
            end_x = request.unlock_end_x
            end_y = request.unlock_end_y
            await run_adb(
                "shell", "input", "swipe",
                str(start_x), str(start_y), str(end_x), str(end_y), str(duration),
                serial=device_serial
            )
        else:
            # 长按解锁（swipe 同一点）
            await run_adb(
                "shell", "input", "swipe",
                str(start_x), str(start_y), str(start_x), str(start_y), str(duration),
                serial=device_serial
            )

        return TestResult(success=True, message="解锁命令已发送")
    except Exception as e:
        return TestResult(success=False, message=f"解锁失败: {str(e)}")


@router.post("/{device_serial}/lock", response_model=TestResult)
async def lock_screen(device_serial: str):
    """锁定设备屏幕"""
    try:
        # 使用 KEYCODE_POWER 锁定屏幕（如果屏幕亮着会关闭屏幕）
        # 或者使用 KEYCODE_SLEEP 直接进入睡眠状态
        await run_adb("shell", "input", "keyevent", "KEYCODE_SLEEP", serial=device_serial)
        return TestResult(success=True, message="锁屏命令已发送")
    except Exception as e:
        return TestResult(success=False, message=f"锁屏失败: {str(e)}")
