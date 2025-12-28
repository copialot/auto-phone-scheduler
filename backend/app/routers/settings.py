from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
import httpx

from app.database import get_db
from app.models.settings import SystemSettings

router = APIRouter(prefix="/api/settings", tags=["settings"])


class SettingsResponse(BaseModel):
    autoglm_base_url: str | None = None
    autoglm_api_key: str | None = None
    autoglm_model: str | None = None
    autoglm_max_steps: int | None = None
    selected_device: str | None = None  # 用户选择的设备 serial


class SettingsUpdate(BaseModel):
    autoglm_base_url: str | None = None
    autoglm_api_key: str | None = None
    autoglm_model: str | None = None
    autoglm_max_steps: int | None = None
    selected_device: str | None = None  # 用户选择的设备 serial


SETTINGS_KEYS = [
    "autoglm_base_url",
    "autoglm_api_key",
    "autoglm_model",
    "autoglm_max_steps",
    "selected_device",
]


@router.get("", response_model=SettingsResponse)
async def get_settings(db: AsyncSession = Depends(get_db)):
    """获取系统设置"""
    result = await db.execute(
        select(SystemSettings).where(SystemSettings.key.in_(SETTINGS_KEYS))
    )
    settings = {s.key: s.value for s in result.scalars().all()}

    return SettingsResponse(
        autoglm_base_url=settings.get("autoglm_base_url"),
        autoglm_api_key=_mask_api_key(settings.get("autoglm_api_key")),
        autoglm_model=settings.get("autoglm_model"),
        autoglm_max_steps=int(settings.get("autoglm_max_steps", 100)),
        selected_device=settings.get("selected_device"),
    )


@router.put("", response_model=SettingsResponse)
async def update_settings(
    settings_in: SettingsUpdate, db: AsyncSession = Depends(get_db)
):
    """更新系统设置"""
    update_data = settings_in.model_dump(exclude_unset=True)

    for key, value in update_data.items():
        if value is None:
            continue

        result = await db.execute(
            select(SystemSettings).where(SystemSettings.key == key)
        )
        setting = result.scalar_one_or_none()

        if setting:
            setting.value = str(value)
        else:
            setting = SystemSettings(key=key, value=str(value))
            db.add(setting)

    await db.commit()

    return await get_settings(db)


def _mask_api_key(api_key: str | None) -> str | None:
    """隐藏 API Key 中间部分"""
    if not api_key:
        return None
    if len(api_key) <= 8:
        return "*" * len(api_key)
    return api_key[:4] + "*" * (len(api_key) - 8) + api_key[-4:]


class TestResult(BaseModel):
    success: bool
    message: str
    models: list[str] | None = None


@router.post("/test", response_model=TestResult)
async def test_model_connection(db: AsyncSession = Depends(get_db)):
    """测试大模型连接是否正常"""
    # 获取当前设置
    result = await db.execute(
        select(SystemSettings).where(SystemSettings.key.in_(SETTINGS_KEYS))
    )
    settings = {s.key: s.value for s in result.scalars().all()}

    base_url = settings.get("autoglm_base_url")
    api_key = settings.get("autoglm_api_key")

    if not base_url or not api_key:
        return TestResult(
            success=False,
            message="请先配置 API 地址和 API Key",
        )

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            # 尝试获取模型列表
            response = await client.get(
                f"{base_url}/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )

            if response.status_code == 200:
                data = response.json()
                models = [m.get("id", m.get("name", "unknown")) for m in data.get("data", [])]
                return TestResult(
                    success=True,
                    message="连接成功",
                    models=models[:10],  # 最多返回10个模型
                )
            elif response.status_code == 401:
                return TestResult(
                    success=False,
                    message="API Key 无效或已过期",
                )
            else:
                return TestResult(
                    success=False,
                    message=f"请求失败: HTTP {response.status_code}",
                )
    except httpx.TimeoutException:
        return TestResult(
            success=False,
            message="连接超时，请检查网络或 API 地址",
        )
    except httpx.ConnectError:
        return TestResult(
            success=False,
            message="无法连接到 API 服务器，请检查 API 地址",
        )
    except Exception as e:
        return TestResult(
            success=False,
            message=f"测试失败: {str(e)}",
        )
