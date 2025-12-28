from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.database import init_db
from app.routers import (
    tasks_router,
    executions_router,
    notifications_router,
    devices_router,
    settings_router,
    system_prompts_router,
    task_templates_router,
    debug_router,
    app_packages_router,
    device_configs_router,
)
from app.services.scheduler import SchedulerService
from app.services.socket_manager import create_socket_app
from app.config import get_settings
from app.patches import apply_all_patches
from app.patches.phone_agent_patch import load_custom_app_packages

settings = get_settings()

# Apply patches to third-party libraries
apply_all_patches()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时
    await init_db()
    # 加载自定义 APP 包名
    await load_custom_app_packages()
    scheduler = SchedulerService.get_instance()
    await scheduler.start()
    yield
    # 关闭时
    await scheduler.shutdown()


app = FastAPI(
    title="AutoGLM 定时任务执行器",
    description="基于 AutoGLM 的手机自动化定时任务系统",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # 使用 * 时不能启用 credentials
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(tasks_router)
app.include_router(executions_router)
app.include_router(notifications_router)
app.include_router(devices_router)
app.include_router(settings_router)
app.include_router(system_prompts_router)
app.include_router(task_templates_router)
app.include_router(debug_router)
app.include_router(app_packages_router)
app.include_router(device_configs_router)

# 静态文件 - 录屏
app.mount("/recordings", StaticFiles(directory=settings.recordings_dir), name="recordings")


@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {"status": "ok"}


# 包装 Socket.IO
socket_app = create_socket_app(app)


if __name__ == "__main__":
    import uvicorn

    # 使用 socket_app 而不是 app，以支持 Socket.IO
    uvicorn.run(socket_app, host="0.0.0.0", port=8000)
