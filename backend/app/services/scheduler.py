import asyncio
import json
import random
import re
import queue
import threading
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import async_session
from app.models.task import Task
from app.models.execution import Execution, ExecutionStatus
from app.models.notification import NotificationChannel
from app.services.adb import run_adb
from app.services.autoglm import AutoGLMService
from app.services.recorder import RecorderService
from app.services.notifier import NotifierService, NotificationType
from app.services.execution_events import event_bus


def parse_action_to_object(action: Any) -> dict[str, Any] | None:
    """
    解析 action 为对象，支持多种格式：
    - JSON 对象: {"action": "Tap", "x": 100}
    - 函数调用: Tap(x=100, y=200)
    - 简单名称: Finish 或 [finish]
    """
    if not action:
        return None

    # 已经是字典
    if isinstance(action, dict):
        return action

    # 字符串格式
    if isinstance(action, str):
        action_str = action.strip()
        if not action_str:
            return None

        # 格式1: JSON 对象 {"action": "Tap", "x": 100}
        json_match = re.search(r'\{[\s\S]*?"action"[\s\S]*?\}', action_str, re.IGNORECASE)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        # 格式2: 函数调用格式 Tap(x=100, y=200) 或 Launch(package="com.xxx")
        func_match = re.match(r'^(\w+)\((.*)\)$', action_str, re.DOTALL)
        if func_match:
            action_name = func_match.group(1)
            params_str = func_match.group(2)
            result: dict[str, Any] = {"action": action_name}

            # 解析参数
            param_regex = re.compile(r'(\w+)=("([^"]*)"|\'([^\']*)\'|(\d+)|(\w+))')
            for match in param_regex.finditer(params_str):
                key = match.group(1)
                # 优先使用双引号内容，其次单引号，然后数字，最后标识符
                value: Any = match.group(3) or match.group(4)
                if value is None:
                    if match.group(5):
                        value = int(match.group(5))
                    else:
                        value = match.group(6)
                result[key] = value

            return result

        # 格式3: 简单的动作名称 如 Finish 或 [finish]
        simple_match = re.match(r'^\[?(\w+)\]?$', action_str, re.IGNORECASE)
        if simple_match:
            return {"action": simple_match.group(1)}

    return None

if TYPE_CHECKING:
    from apscheduler.job import Job


class SchedulerService:
    """任务调度服务"""

    _instance: "SchedulerService | None" = None

    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.autoglm = AutoGLMService()
        self.notifier = NotifierService()

    @classmethod
    def get_instance(cls) -> "SchedulerService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def start(self):
        """启动调度器并加载已有任务"""
        if not self.scheduler.running:
            self.scheduler.start()
            await self._cleanup_stale_executions()
            await self.load_tasks()

    async def _cleanup_stale_executions(self):
        """清理启动时残留的 RUNNING 状态执行记录（服务器异常重启后的清理）"""
        async with async_session() as session:
            result = await session.execute(
                select(Execution).where(Execution.status == ExecutionStatus.RUNNING)
            )
            stale_executions = result.scalars().all()

            if stale_executions:
                for execution in stale_executions:
                    execution.status = ExecutionStatus.FAILED
                    execution.finished_at = datetime.utcnow()
                    execution.error_message = "服务器重启，任务被中断"
                await session.commit()
                print(f"[Scheduler] 清理了 {len(stale_executions)} 个残留的执行记录")

    async def shutdown(self):
        """关闭调度器"""
        if self.scheduler.running:
            self.scheduler.shutdown()

    async def load_tasks(self):
        """从数据库加载所有启用的任务"""
        async with async_session() as session:
            result = await session.execute(
                select(Task).where(Task.enabled == True)
            )
            tasks = result.scalars().all()
            for task in tasks:
                self.add_job(task)

    def add_job(self, task: Task):
        """添加定时任务"""
        job_id = f"task_{task.id}"

        # 移除已存在的任务
        if self.scheduler.get_job(job_id):
            self.scheduler.remove_job(job_id)

        if task.enabled:
            # 解析 cron 表达式，使用任务指定的时区或服务器本地时区
            cron_parts = task.cron_expression.split()
            if task.timezone:
                try:
                    tz = ZoneInfo(task.timezone)
                except Exception:
                    # 无效时区，回退到服务器本地时区
                    tz = datetime.now().astimezone().tzinfo
            else:
                tz = datetime.now().astimezone().tzinfo
            trigger = CronTrigger(
                minute=cron_parts[0] if len(cron_parts) > 0 else "*",
                hour=cron_parts[1] if len(cron_parts) > 1 else "*",
                day=cron_parts[2] if len(cron_parts) > 2 else "*",
                month=cron_parts[3] if len(cron_parts) > 3 else "*",
                day_of_week=cron_parts[4] if len(cron_parts) > 4 else "*",
                timezone=tz,
            )

            # 如果设置了随机延迟，使用包装器函数
            if task.random_delay_minutes and task.random_delay_minutes > 0:
                self.scheduler.add_job(
                    self._schedule_delayed_task,
                    trigger,
                    id=job_id,
                    args=[task.id, task.random_delay_minutes],
                    replace_existing=True,
                )
            else:
                self.scheduler.add_job(
                    self.execute_task,
                    trigger,
                    id=job_id,
                    args=[task.id],
                    replace_existing=True,
                )

    async def _schedule_delayed_task(self, task_id: int, max_delay_minutes: int):
        """调度延迟执行的任务"""
        # 随机选择 0 到 max_delay_minutes 之间的延迟时间
        delay_minutes = random.randint(0, max_delay_minutes)
        run_time = datetime.now() + timedelta(minutes=delay_minutes)

        # 创建一次性任务
        delayed_job_id = f"task_{task_id}_delayed_{run_time.timestamp()}"
        self.scheduler.add_job(
            self.execute_task,
            DateTrigger(run_date=run_time),
            id=delayed_job_id,
            args=[task_id],
            replace_existing=True,
        )

    def remove_job(self, task_id: int):
        """移除定时任务"""
        job_id = f"task_{task_id}"
        if self.scheduler.get_job(job_id):
            self.scheduler.remove_job(job_id)

    def get_next_run_time(self, task_id: int) -> datetime | None:
        """获取下次执行时间"""
        job_id = f"task_{task_id}"
        job = self.scheduler.get_job(job_id)
        if job:
            return job.next_run_time
        return None

    async def _get_all_devices(self) -> list[tuple[str, str, str | None]]:
        """获取所有已连接的设备信息

        Returns:
            list of (serial, status, model)
        """
        devices = []
        try:
            stdout, _ = await run_adb("devices", "-l")
            output = stdout.decode()

            lines = output.strip().split("\n")[1:]  # 跳过标题行
            for line in lines:
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    serial = parts[0]
                    status = parts[1]
                    model = None
                    for part in parts[2:]:
                        if part.startswith("model:"):
                            model = part.split(":")[1]
                            break
                    devices.append((serial, status, model))
        except Exception:
            pass
        return devices

    async def _get_selected_device(
        self, session: "AsyncSession", task_device_serial: str | None = None
    ) -> tuple[str | None, str | None, str | None]:
        """获取任务执行设备

        优先级：任务指定设备 > 全局设置设备 > 第一个在线设备
        如果指定的设备不可用则直接失败，不回退

        Args:
            session: 数据库会话
            task_device_serial: 任务指定的设备 serial

        Returns:
            (serial, model, error_msg) 元组
            - 成功时返回 (serial, model, None)
            - 失败时返回 (None, None, error_msg)
        """
        from sqlalchemy import select as sql_select
        from app.models.settings import SystemSettings

        # 获取所有在线设备
        devices = await self._get_all_devices()
        online_devices = [(s, m) for s, status, m in devices if status == "device"]

        if not online_devices:
            return None, None, "未找到可用设备"

        # 1. 优先使用任务指定的设备
        if task_device_serial:
            for serial, model in online_devices:
                if serial == task_device_serial:
                    return serial, model, None
            return None, None, f"任务指定设备 {task_device_serial} 不可用"

        # 2. 其次使用全局设置的设备
        result = await session.execute(
            sql_select(SystemSettings).where(SystemSettings.key == "selected_device")
        )
        setting = result.scalar_one_or_none()
        global_serial = setting.value if setting else None

        if global_serial:
            for serial, model in online_devices:
                if serial == global_serial:
                    return serial, model, None
            return None, None, f"全局设置设备 {global_serial} 不可用"

        # 3. 回退到第一个在线设备
        return online_devices[0][0], online_devices[0][1], None

    async def _get_first_device(self) -> tuple[str | None, str | None]:
        """获取第一个已连接的设备信息（兼容旧调用）"""
        devices = await self._get_all_devices()
        for serial, status, model in devices:
            if status == "device":
                return serial, model
        return None, None

    async def _is_device_busy(
        self, session: AsyncSession, device_serial: str, exclude_execution_id: int | None = None
    ) -> tuple[bool, int | None]:
        """检查设备是否被其他任务占用

        Args:
            session: 数据库会话
            device_serial: 设备序列号
            exclude_execution_id: 排除的执行记录ID（用于当前任务自己的检查）

        Returns:
            (is_busy, running_execution_id) - 如果设备被占用返回 (True, execution_id)
        """
        query = select(Execution).where(
            Execution.device_serial == device_serial,
            Execution.status == ExecutionStatus.RUNNING,
        )
        if exclude_execution_id:
            query = query.where(Execution.id != exclude_execution_id)

        # 使用 first() 而不是 scalar_one_or_none()，以处理可能存在多个运行中任务的情况
        result = await session.execute(query.limit(1))
        running_execution = result.scalar_one_or_none()

        if running_execution:
            return True, running_execution.id
        return False, None

    async def _is_screen_locked(self, device_serial: str) -> tuple[bool, bool]:
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
            # 如果检测失败，假设需要唤醒和解锁
            return False, True

    async def _wake_and_unlock_device(
        self, session: AsyncSession, device_serial: str, wake: bool, unlock: bool,
        logs: list[dict] | None = None
    ) -> tuple[bool, str | None]:
        """唤醒和解锁设备（会先检测屏幕状态，避免误操作）

        Args:
            session: 数据库会话
            device_serial: 设备序列号
            wake: 是否唤醒
            unlock: 是否解锁
            logs: 可选的日志列表，用于收集操作日志

        Returns:
            (success, error_message) - 成功时返回 (True, None)，失败时返回 (False, error_message)
        """
        from app.models.device_config import DeviceConfig

        def add_log(message: str, log_type: str = "info"):
            if logs is not None:
                logs.append({
                    "step": 0,
                    "thinking": message,
                    "action": {"action": "SystemLog", "type": log_type},
                    "timestamp": datetime.utcnow().isoformat(),
                })

        if not wake and not unlock:
            return True, None

        # 获取设备配置
        result = await session.execute(
            select(DeviceConfig).where(DeviceConfig.device_serial == device_serial)
        )
        config = result.scalar_one_or_none()

        # 先检测屏幕状态
        screen_on, is_locked = await self._is_screen_locked(device_serial)
        add_log(f"检测屏幕状态: {'亮屏' if screen_on else '息屏'}, {'已锁定' if is_locked else '未锁定'}")

        # 唤醒设备（仅在屏幕未亮时）
        if wake and not screen_on:
            add_log("正在唤醒设备...")
            if config and config.wake_enabled:
                # 使用自定义唤醒命令或默认命令
                if config.wake_command:
                    await run_adb("shell", config.wake_command, serial=device_serial)
                else:
                    await run_adb("shell", "input", "keyevent", "KEYCODE_WAKEUP", serial=device_serial)
            else:
                # 默认唤醒命令
                await run_adb("shell", "input", "keyevent", "KEYCODE_WAKEUP", serial=device_serial)
            # 等待屏幕亮起
            await asyncio.sleep(0.5)
            # 重新检测状态
            screen_on, is_locked = await self._is_screen_locked(device_serial)
            add_log(f"唤醒完成，屏幕状态: {'亮屏' if screen_on else '息屏'}")

        # 解锁设备（仅在锁屏状态时）
        if unlock and is_locked and config and config.unlock_enabled and config.unlock_type:
            add_log("正在解锁设备...")
            max_retries = 3
            for attempt in range(max_retries):
                if config.unlock_type in ("swipe", "longpress"):
                    start_x = config.unlock_start_x or 0
                    start_y = config.unlock_start_y or 0
                    end_x = config.unlock_end_x or start_x
                    end_y = config.unlock_end_y or start_y
                    duration = config.unlock_duration or 300

                    await run_adb(
                        "shell", "input", "swipe",
                        str(start_x), str(start_y), str(end_x), str(end_y), str(duration),
                        serial=device_serial
                    )

                # 等待解锁完成
                await asyncio.sleep(0.5)

                # 检查是否已解锁
                _, still_locked = await self._is_screen_locked(device_serial)
                if not still_locked:
                    # 解锁成功
                    add_log("设备解锁成功")
                    return True, None

                # 如果还是锁定状态，等待后重试
                if attempt < max_retries - 1:
                    add_log(f"解锁未成功，正在重试 ({attempt + 2}/{max_retries})...")
                    await asyncio.sleep(0.5)

            # 重试 3 次仍然锁定，返回失败
            error_msg = f"设备解锁失败：重试 {max_retries} 次后仍处于锁屏状态"
            add_log(error_msg, "error")
            return False, error_msg

        return True, None

    async def execute_task(self, task_id: int):
        """执行任务（定时任务调用），使用与 run_task_with_execution 相同的配置加载逻辑"""
        from sqlalchemy import select as sql_select
        from app.models.settings import SystemSettings
        from app.config import get_settings

        settings = get_settings()

        async with async_session() as session:
            # 获取任务信息
            result = await session.execute(select(Task).where(Task.id == task_id))
            task = result.scalar_one_or_none()
            if not task:
                return

            # 从数据库加载设置
            result = await session.execute(sql_select(SystemSettings))
            db_settings = {s.key: s.value for s in result.scalars().all()}

            # 获取设备信息（优先使用任务指定设备）
            device_serial, device_model, device_error = await self._get_selected_device(
                session, task_device_serial=task.device_serial
            )

            # 检查设备是否可用
            if not device_serial:
                # 创建失败的执行记录
                execution = Execution(
                    task_id=task_id,
                    status=ExecutionStatus.FAILED,
                    started_at=datetime.utcnow(),
                    finished_at=datetime.utcnow(),
                    error_message=device_error,
                )
                session.add(execution)
                await session.commit()
                await session.refresh(execution)
                # 发送失败通知
                await self._send_notifications(session, task, execution)
                return

            # 检查设备是否被其他任务占用
            is_busy, running_exec_id = await self._is_device_busy(session, device_serial)
            if is_busy:
                # 创建失败的执行记录
                execution = Execution(
                    task_id=task_id,
                    status=ExecutionStatus.FAILED,
                    started_at=datetime.utcnow(),
                    finished_at=datetime.utcnow(),
                    device_serial=device_serial,
                    error_message=f"设备 {device_serial} 正在被其他任务占用（执行记录 #{running_exec_id}）",
                )
                session.add(execution)
                await session.commit()
                await session.refresh(execution)
                # 发送失败通知
                await self._send_notifications(session, task, execution)
                return

            # 创建执行记录（包含设备信息）
            execution = Execution(
                task_id=task_id,
                status=ExecutionStatus.RUNNING,
                started_at=datetime.utcnow(),
                device_serial=device_serial,  # 保存实际使用的设备
            )
            session.add(execution)
            await session.commit()
            await session.refresh(execution)

            # 执行前唤醒和解锁设备（收集日志）
            wake_unlock_logs: list[dict] = []
            unlock_success, unlock_error = await self._wake_and_unlock_device(
                session, device_serial,
                wake=task.wake_before_run,
                unlock=task.unlock_before_run,
                logs=wake_unlock_logs
            )

            # 保存唤醒解锁日志到执行记录，并发布到 SSE
            if wake_unlock_logs:
                execution.steps = wake_unlock_logs.copy()
                await session.commit()
                # 发布唤醒解锁日志到 SSE（实时显示）
                for log in wake_unlock_logs:
                    await event_bus.publish(execution.id, "step", log)

            # 检查解锁是否成功
            if not unlock_success:
                execution.status = ExecutionStatus.FAILED
                execution.finished_at = datetime.utcnow()
                execution.error_message = unlock_error
                await session.commit()
                # 发布失败事件到 SSE
                await event_bus.publish(execution.id, "done", {
                    "success": False,
                    "message": unlock_error,
                })
                # 发送失败通知
                await self._send_notifications(session, task, execution)
                return

            base_url = db_settings.get("autoglm_base_url") or settings.autoglm_base_url
            api_key = db_settings.get("autoglm_api_key") or settings.autoglm_api_key
            model = db_settings.get("autoglm_model") or settings.autoglm_model
            # 优先使用任务专属的 max_steps，否则使用全局设置
            global_max_steps = int(db_settings.get("autoglm_max_steps") or settings.autoglm_max_steps)
            max_steps = task.max_steps if task.max_steps is not None else global_max_steps
            lang = settings.autoglm_lang

            # 获取系统提示词规则
            system_prompt, prefix_prompt, suffix_prompt = await self.autoglm.get_system_prompts(
                session, device_serial, device_model
            )
            cmd = self.autoglm.apply_prompt_rules(task.command, prefix_prompt, suffix_prompt)

            # 创建带设备 serial 的录制服务
            recorder = RecorderService(device_serial=device_serial)

            try:
                # 开始录屏
                recording_path = await recorder.start_recording(execution.id)

                # 直接使用 PhoneAgent
                from phone_agent import PhoneAgent
                from phone_agent.model import ModelConfig
                from phone_agent.agent import AgentConfig
                from phone_agent.config import get_system_prompt

                # 使用队列实现实时步骤更新
                step_queue: queue.Queue = queue.Queue()
                result_holder = {"success": False, "error_msg": None}

                def run_agent_sync():
                    """在线程池中同步运行 agent，通过队列传递步骤"""
                    model_config = ModelConfig(
                        base_url=base_url,
                        api_key=api_key,
                        model_name=model,
                        max_tokens=3000,
                        temperature=0.0,
                        top_p=0.85,
                        frequency_penalty=0.2,
                    )

                    final_system_prompt = get_system_prompt(lang)
                    if system_prompt:
                        final_system_prompt = f"{final_system_prompt}\n\n# 额外规则\n{system_prompt}"

                    agent_config = AgentConfig(
                        max_steps=max_steps,
                        device_id=device_serial,
                        lang=lang,
                        system_prompt=final_system_prompt,
                        verbose=True,
                    )

                    # 自定义回调函数，避免使用默认的 input() 阻塞
                    # 定时任务默认自动确认敏感操作
                    def auto_confirm_callback(message: str) -> bool:
                        return task.auto_confirm_sensitive

                    def noop_takeover_callback(message: str) -> None:
                        pass

                    agent = PhoneAgent(
                        model_config=model_config,
                        agent_config=agent_config,
                        takeover_callback=noop_takeover_callback,
                        confirmation_callback=auto_confirm_callback,
                    )

                    try:
                        step_result = agent.step(cmd)

                        while True:
                            # 确保 thinking 是字符串
                            thinking_str = str(step_result.thinking) if step_result.thinking else ""

                            # 使用通用解析函数解析 action
                            action_obj = parse_action_to_object(step_result.action)
                            action_str = str(step_result.action) if step_result.action else ""

                            step_info = {
                                "step": agent.step_count,
                                "thinking": thinking_str,
                                "action": action_obj or action_str,
                                "description": f"<thinking>{thinking_str}</thinking>\n<answer>{action_str}</answer>" if thinking_str else action_str,
                                "timestamp": datetime.utcnow().isoformat(),
                            }
                            # 将步骤放入队列
                            step_queue.put(("step", step_info))

                            if step_result.finished:
                                result_holder["success"] = step_result.success
                                break

                            if agent.step_count >= max_steps:
                                result_holder["error_msg"] = "已达到最大步数限制"
                                break

                            step_result = agent.step()

                    except Exception as e:
                        result_holder["error_msg"] = str(e)
                    finally:
                        agent.reset()
                        # 发送完成信号
                        step_queue.put(("done", None))

                # 启动 agent 线程
                agent_thread = threading.Thread(target=run_agent_sync)
                agent_thread.start()

                # 在主协程中处理队列，实时更新数据库
                # 使用非阻塞方式检查队列，确保不阻塞事件循环
                # 将唤醒解锁日志作为初始步骤
                collected_steps = wake_unlock_logs.copy()
                loop = asyncio.get_event_loop()
                while True:
                    # 使用 run_in_executor 将阻塞的 queue.get 放到线程池
                    # 这样不会阻塞事件循环
                    try:
                        msg = await asyncio.wait_for(
                            loop.run_in_executor(None, lambda: step_queue.get(timeout=0.5)),
                            timeout=1.0
                        )
                        msg_type, data = msg
                        if msg_type == "done":
                            # 发布完成事件
                            await event_bus.publish(execution.id, "done", {
                                "success": result_holder["success"],
                                "message": result_holder.get("error_msg"),
                            })
                            break
                        elif msg_type == "step":
                            collected_steps.append(data)
                            # 实时更新数据库
                            execution.steps = collected_steps.copy()
                            await session.commit()
                            # 发布步骤事件
                            await event_bus.publish(execution.id, "step", data)
                    except (queue.Empty, asyncio.TimeoutError):
                        # 队列空或超时，让出控制权给其他协程
                        await asyncio.sleep(0)
                        continue

                # 等待线程完成
                agent_thread.join(timeout=5)
                success = result_holder["success"]
                error_msg = result_holder["error_msg"]

                # 停止录屏并获取实际的录制文件路径
                actual_recording_path = await recorder.stop_recording()

                # 执行完成后返回主屏幕
                if task.go_home_after_run:
                    try:
                        await run_adb("shell", "input", "keyevent", "KEYCODE_HOME", serial=device_serial)
                    except Exception:
                        pass  # 忽略返回主屏幕的错误

                # 更新执行记录
                execution.status = ExecutionStatus.SUCCESS if success else ExecutionStatus.FAILED
                execution.finished_at = datetime.utcnow()
                execution.steps = collected_steps
                # 使用实际的录制路径（stop_recording 会验证文件是否存在）
                execution.recording_path = actual_recording_path
                if error_msg:
                    execution.error_message = error_msg

                await session.commit()

                # 发送通知
                await self._send_notifications(session, task, execution)

            except Exception as e:
                # 停止录屏
                await recorder.stop_recording()

                # 执行失败也尝试返回主屏幕
                if task.go_home_after_run:
                    try:
                        await run_adb("shell", "input", "keyevent", "KEYCODE_HOME", serial=device_serial)
                    except Exception:
                        pass

                # 更新执行记录为失败
                execution.status = ExecutionStatus.FAILED
                execution.finished_at = datetime.utcnow()
                execution.error_message = str(e)
                await session.commit()

                # 发送失败通知
                await self._send_notifications(session, task, execution)

    async def _send_notifications(
        self, session: AsyncSession, task: Task, execution: Execution
    ):
        """发送通知"""
        should_notify = (
            (execution.status == ExecutionStatus.SUCCESS and task.notify_on_success)
            or (execution.status == ExecutionStatus.FAILED and task.notify_on_failure)
        )

        if not should_notify:
            return

        # 获取通知渠道
        # 如果任务指定了通知渠道，只使用指定的渠道
        # 否则使用所有启用的渠道
        if task.notification_channel_ids:
            result = await session.execute(
                select(NotificationChannel).where(
                    NotificationChannel.enabled == True,
                    NotificationChannel.id.in_(task.notification_channel_ids),
                )
            )
        else:
            result = await session.execute(
                select(NotificationChannel).where(NotificationChannel.enabled == True)
            )
        channels = result.scalars().all()

        status_text = "成功" if execution.status == ExecutionStatus.SUCCESS else "失败"
        title = f"任务执行{status_text}: {task.name}"
        content = f"- 任务: {task.name}\n- 状态: {status_text}\n- 时间: {execution.finished_at}"

        # 添加最后一步的结果消息
        if execution.steps and len(execution.steps) > 0:
            last_step = execution.steps[-1]
            # 从 action 中提取 message（finish action 的结果消息）
            last_action = last_step.get("action")
            if isinstance(last_action, dict):
                result_msg = last_action.get("message")
                if result_msg:
                    content += f"\n- 结果: {result_msg}"
            elif isinstance(last_action, str) and "message=" in last_action:
                # 解析字符串格式的 message
                import re
                msg_match = re.search(r'message="([^"]*)"', last_action)
                if msg_match:
                    content += f"\n- 结果: {msg_match.group(1)}"

        if execution.error_message:
            content += f"\n- 错误: {execution.error_message}"

        for channel in channels:
            await self.notifier.send_notification(
                NotificationType(channel.type),
                channel.config,
                title,
                content,
            )

    async def run_task_now(self, task_id: int):
        """立即执行任务"""
        await self.execute_task(task_id)

    async def run_task_with_execution(self, task_id: int, execution_id: int):
        """执行任务（使用已创建的执行记录），实时保存步骤到数据库，支持流式输出"""
        from sqlalchemy import select as sql_select
        from app.models.settings import SystemSettings
        from app.config import get_settings
        import threading
        import queue

        settings = get_settings()
        steps_list = []  # 用于收集步骤
        step_queue: queue.Queue = queue.Queue()  # 线程安全的步骤队列
        token_queue: queue.Queue = queue.Queue()  # 流式 token 队列
        stop_event = threading.Event()  # 停止信号

        async def save_step(step_info: dict):
            """实时保存步骤到数据库"""
            steps_list.append(step_info)
            async with async_session() as step_session:
                result = await step_session.execute(
                    sql_select(Execution).where(Execution.id == execution_id)
                )
                exec_record = result.scalar_one_or_none()
                if exec_record:
                    exec_record.steps = steps_list.copy()
                    await step_session.commit()

        async def event_publisher_task():
            """后台任务：从队列读取事件并发布SSE（非阻塞）"""
            loop = asyncio.get_event_loop()

            while not stop_event.is_set() or not step_queue.empty() or not token_queue.empty():
                # 优先处理 token 流（打字机效果）
                try:
                    token_msg = await asyncio.wait_for(
                        loop.run_in_executor(None, lambda: token_queue.get(timeout=0.05)),
                        timeout=0.1
                    )
                    msg_type, data = token_msg
                    if msg_type == "token":
                        phase, content, step_num = data
                        await event_bus.publish(execution_id, "token", {
                            "step": step_num,
                            "phase": phase,
                            "content": content,
                        })
                    continue
                except (queue.Empty, asyncio.TimeoutError):
                    pass

                # 处理完整步骤
                try:
                    step_info = await asyncio.wait_for(
                        loop.run_in_executor(None, lambda: step_queue.get(timeout=0.1)),
                        timeout=0.15
                    )
                    await save_step(step_info)
                    # 发布步骤完成事件
                    await event_bus.publish(execution_id, "step", step_info)
                except (queue.Empty, asyncio.TimeoutError):
                    await asyncio.sleep(0)
                    continue

        async with async_session() as session:
            # 获取任务信息
            result = await session.execute(select(Task).where(Task.id == task_id))
            task = result.scalar_one_or_none()
            if not task:
                return

            # 获取执行记录
            result = await session.execute(
                select(Execution).where(Execution.id == execution_id)
            )
            execution = result.scalar_one_or_none()
            if not execution:
                return

            # 从数据库加载设置（参考 debug.py 的方式）
            result = await session.execute(sql_select(SystemSettings))
            db_settings = {s.key: s.value for s in result.scalars().all()}

            # 获取设备信息（优先使用任务指定设备）
            device_serial, device_model, device_error = await self._get_selected_device(
                session, task_device_serial=task.device_serial
            )

            # 更新执行记录的设备信息
            execution.device_serial = device_serial
            await session.commit()

            # 检查设备是否可用
            if not device_serial:
                execution.status = ExecutionStatus.FAILED
                execution.finished_at = datetime.utcnow()
                execution.error_message = device_error
                await session.commit()
                # 发布失败事件到 SSE
                await event_bus.publish(execution_id, "done", {
                    "success": False,
                    "message": device_error,
                })
                # 发送失败通知
                await self._send_notifications(session, task, execution)
                return

            # 检查设备是否被其他任务占用（排除当前执行记录）
            is_busy, running_exec_id = await self._is_device_busy(
                session, device_serial, exclude_execution_id=execution_id
            )
            if is_busy:
                busy_error = f"设备 {device_serial} 正在被其他任务占用（执行记录 #{running_exec_id}）"
                execution.status = ExecutionStatus.FAILED
                execution.finished_at = datetime.utcnow()
                execution.error_message = busy_error
                await session.commit()
                # 发布失败事件到 SSE
                await event_bus.publish(execution_id, "done", {
                    "success": False,
                    "message": busy_error,
                })
                # 发送失败通知
                await self._send_notifications(session, task, execution)
                return

            # 执行前唤醒和解锁设备（收集日志）
            wake_unlock_logs: list[dict] = []
            unlock_success, unlock_error = await self._wake_and_unlock_device(
                session, device_serial,
                wake=task.wake_before_run,
                unlock=task.unlock_before_run,
                logs=wake_unlock_logs
            )

            # 保存唤醒解锁日志到执行记录，并发布到 SSE
            if wake_unlock_logs:
                steps_list.extend(wake_unlock_logs)
                execution.steps = steps_list.copy()
                await session.commit()
                # 发布唤醒解锁日志到 SSE
                for log in wake_unlock_logs:
                    await event_bus.publish(execution_id, "step", log)

            # 检查解锁是否成功
            if not unlock_success:
                execution.status = ExecutionStatus.FAILED
                execution.finished_at = datetime.utcnow()
                execution.error_message = unlock_error
                await session.commit()
                # 发布失败事件到 SSE
                await event_bus.publish(execution_id, "done", {
                    "success": False,
                    "message": unlock_error,
                })
                # 发送失败通知
                await self._send_notifications(session, task, execution)
                return

            base_url = db_settings.get("autoglm_base_url") or settings.autoglm_base_url
            api_key = db_settings.get("autoglm_api_key") or settings.autoglm_api_key
            model = db_settings.get("autoglm_model") or settings.autoglm_model
            # 优先使用任务专属的 max_steps，否则使用全局设置
            global_max_steps = int(db_settings.get("autoglm_max_steps") or settings.autoglm_max_steps)
            max_steps = task.max_steps if task.max_steps is not None else global_max_steps
            lang = settings.autoglm_lang

            # 获取系统提示词规则
            system_prompt, prefix_prompt, suffix_prompt = await self.autoglm.get_system_prompts(
                session, device_serial, device_model
            )
            cmd = self.autoglm.apply_prompt_rules(task.command, prefix_prompt, suffix_prompt)

            # 创建带设备 serial 的录制服务
            recorder = RecorderService(device_serial=device_serial)

            try:
                # 开始录屏
                recording_path = await recorder.start_recording(execution.id)

                # 直接使用 PhoneAgent，实现实时步骤保存
                from phone_agent import PhoneAgent
                from phone_agent.model import ModelConfig
                from phone_agent.agent import AgentConfig
                from phone_agent.config import get_system_prompt
                from app.services.streaming_model import patch_phone_agent, unpatch_phone_agent

                # 当前步骤计数（用于 token 回调）
                current_step_holder = {"step": 0}

                def token_callback(phase: str, content: str):
                    """流式 token 回调，将 token 放入队列"""
                    token_queue.put(("token", (phase, content, current_step_holder["step"])))

                def run_agent_sync():
                    """在线程池中同步运行 agent，每步通过队列实时通知"""
                    # 应用流式补丁
                    original_client = patch_phone_agent(token_callback)

                    try:
                        model_config = ModelConfig(
                            base_url=base_url,
                            api_key=api_key,
                            model_name=model,
                            max_tokens=3000,
                            temperature=0.0,
                            top_p=0.85,
                            frequency_penalty=0.2,
                        )

                        final_system_prompt = get_system_prompt(lang)
                        if system_prompt:
                            final_system_prompt = f"{final_system_prompt}\n\n# 额外规则\n{system_prompt}"

                        agent_config = AgentConfig(
                            max_steps=max_steps,
                            device_id=device_serial,
                            lang=lang,
                            system_prompt=final_system_prompt,
                            verbose=True,
                        )

                        # 自定义回调函数，避免使用默认的 input() 阻塞
                        # 定时任务默认自动确认敏感操作
                        def auto_confirm_callback(message: str) -> bool:
                            return task.auto_confirm_sensitive

                        def noop_takeover_callback(message: str) -> None:
                            pass

                        agent = PhoneAgent(
                            model_config=model_config,
                            agent_config=agent_config,
                            takeover_callback=noop_takeover_callback,
                            confirmation_callback=auto_confirm_callback,
                        )

                        success = False
                        error_msg = None

                        try:
                            # 第一步 - 更新步骤计数
                            current_step_holder["step"] = 1
                            step_result = agent.step(cmd)

                            while True:
                                # 确保 thinking 是字符串
                                thinking_str = str(step_result.thinking) if step_result.thinking else ""

                                # 使用通用解析函数解析 action
                                action_obj = parse_action_to_object(step_result.action)
                                action_str = str(step_result.action) if step_result.action else ""

                                # 记录步骤并放入队列（实时保存）
                                step_info = {
                                    "step": agent.step_count,
                                    "thinking": thinking_str,
                                    "action": action_obj or action_str,
                                    "description": f"<thinking>{thinking_str}</thinking>\n<answer>{action_str}</answer>" if thinking_str else action_str,
                                    "timestamp": datetime.utcnow().isoformat(),
                                }
                                step_queue.put(step_info)

                                if step_result.finished:
                                    success = step_result.success
                                    break

                                if agent.step_count >= max_steps:
                                    error_msg = "已达到最大步数限制"
                                    break

                                # 继续下一步 - 更新步骤计数
                                current_step_holder["step"] = agent.step_count + 1
                                step_result = agent.step()

                        except Exception as e:
                            error_msg = str(e)
                        finally:
                            agent.reset()
                            stop_event.set()  # 通知保存任务停止

                        return success, error_msg
                    finally:
                        # 恢复原始 ModelClient
                        unpatch_phone_agent(original_client)

                # 启动事件发布后台任务
                publisher_task = asyncio.create_task(event_publisher_task())

                # 使用线程池执行 agent
                loop = asyncio.get_event_loop()
                success, error_msg = await loop.run_in_executor(None, run_agent_sync)

                # 等待所有事件发布完成
                await publisher_task

                # 停止录屏并获取实际的录制文件路径
                actual_recording_path = await recorder.stop_recording()

                # 执行完成后返回主屏幕
                if task.go_home_after_run:
                    try:
                        await run_adb("shell", "input", "keyevent", "KEYCODE_HOME", serial=device_serial)
                    except Exception:
                        pass  # 忽略返回主屏幕的错误

                # 更新执行记录
                execution.status = ExecutionStatus.SUCCESS if success else ExecutionStatus.FAILED
                execution.finished_at = datetime.utcnow()
                execution.steps = steps_list
                # 使用实际的录制路径（stop_recording 会验证文件是否存在）
                execution.recording_path = actual_recording_path
                if error_msg:
                    execution.error_message = error_msg

                await session.commit()

                # 数据库提交后再发布完成事件到 SSE，确保前端能获取到最新状态
                await event_bus.publish(execution_id, "done", {
                    "success": success,
                    "message": error_msg,
                })

                # 发送通知
                await self._send_notifications(session, task, execution)

            except Exception as e:
                # 确保停止保存任务
                stop_event.set()

                # 停止录屏
                await recorder.stop_recording()

                # 执行失败也尝试返回主屏幕
                if task.go_home_after_run:
                    try:
                        await run_adb("shell", "input", "keyevent", "KEYCODE_HOME", serial=device_serial)
                    except Exception:
                        pass

                # 更新执行记录为失败
                execution.status = ExecutionStatus.FAILED
                execution.finished_at = datetime.utcnow()
                execution.error_message = str(e)
                execution.steps = steps_list  # 保存已收集的步骤
                await session.commit()

                # 数据库提交后再发布失败事件到 SSE，确保前端能获取到最新状态
                await event_bus.publish(execution_id, "done", {
                    "success": False,
                    "message": str(e),
                })

                # 发送失败通知
                await self._send_notifications(session, task, execution)
