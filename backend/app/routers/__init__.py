from app.routers.tasks import router as tasks_router
from app.routers.executions import router as executions_router
from app.routers.notifications import router as notifications_router
from app.routers.devices import router as devices_router
from app.routers.settings import router as settings_router
from app.routers.system_prompts import router as system_prompts_router
from app.routers.task_templates import router as task_templates_router
from app.routers.debug import router as debug_router
from app.routers.app_packages import router as app_packages_router
from app.routers.device_configs import router as device_configs_router

__all__ = [
    "tasks_router",
    "executions_router",
    "notifications_router",
    "devices_router",
    "settings_router",
    "system_prompts_router",
    "task_templates_router",
    "debug_router",
    "app_packages_router",
    "device_configs_router",
]
