"""调试接口 - 即时执行指令（流式）"""
import asyncio
import json
import queue
import re
import threading
import time
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db
from app.config import get_settings
from app.models.settings import SystemSettings
from app.models.execution import Execution, ExecutionStatus
from app.routers.devices import get_connected_devices
from app.services.autoglm import AutoGLMService
from app.services.streaming_model import patch_phone_agent, unpatch_phone_agent

router = APIRouter(prefix="/api/debug", tags=["debug"])
settings = get_settings()


class ExecuteRequest(BaseModel):
    command: str


@router.post("/execute-stream")
async def execute_stream(request: ExecuteRequest, db: AsyncSession = Depends(get_db)):
    """流式执行指令（SSE）- 支持真正的打字机效果"""
    command = request.command
    if not command.strip():
        raise HTTPException(status_code=400, detail="指令不能为空")

    # 从数据库加载设置
    result = await db.execute(select(SystemSettings))
    db_settings = {s.key: s.value for s in result.scalars().all()}

    # 获取已连接设备
    devices = await get_connected_devices()
    online_devices = [d for d in devices if d.status == "device"]
    if not online_devices:
        raise HTTPException(status_code=400, detail="未找到已连接的设备")

    # 优先使用用户选定的设备（严格模式：选定设备不可用则失败）
    selected_serial = db_settings.get("selected_device")
    active_device = None
    if selected_serial:
        active_device = next((d for d in online_devices if d.serial == selected_serial), None)
        if not active_device:
            raise HTTPException(status_code=400, detail=f"指定设备 {selected_serial} 不可用")
    else:
        # 未选择设备，使用第一个在线设备
        active_device = online_devices[0]

    # 检查设备是否被其他任务占用
    running_result = await db.execute(
        select(Execution).where(
            Execution.device_serial == active_device.serial,
            Execution.status == ExecutionStatus.RUNNING,
        ).limit(1)
    )
    running_execution = running_result.scalar_one_or_none()
    if running_execution:
        raise HTTPException(
            status_code=400,
            detail=f"设备 {active_device.serial} 正在被其他任务占用（执行记录 #{running_execution.id}）"
        )

    base_url = db_settings.get("autoglm_base_url") or settings.autoglm_base_url
    api_key = db_settings.get("autoglm_api_key") or settings.autoglm_api_key
    model = db_settings.get("autoglm_model") or settings.autoglm_model
    max_steps = int(db_settings.get("autoglm_max_steps") or settings.autoglm_max_steps)
    lang = settings.autoglm_lang
    device_serial = active_device.serial
    device_model = active_device.model

    # 获取系统提示词规则
    autoglm_service = AutoGLMService()
    system_prompt, prefix_prompt, suffix_prompt = await autoglm_service.get_system_prompts(
        db, device_serial, device_model
    )
    # 应用前后缀规则
    cmd = autoglm_service.apply_prompt_rules(command.strip(), prefix_prompt, suffix_prompt)

    # 使用队列实现线程间通信
    event_queue: queue.Queue = queue.Queue()
    stop_event = threading.Event()

    # 当前步骤计数（用于 token 回调）
    current_step_holder = {"step": 0}

    def token_callback(phase: str, content: str):
        """流式 token 回调，将 token 放入队列"""
        event_queue.put(("token", {
            "type": "token",
            "step": current_step_holder["step"],
            "phase": phase,
            "content": content,
        }))

    def run_agent():
        """在后台线程中运行 agent"""
        from phone_agent import PhoneAgent
        from phone_agent.model import ModelConfig
        from phone_agent.agent import AgentConfig
        from phone_agent.config import get_system_prompt

        # 应用流式补丁
        original_client = patch_phone_agent(token_callback)

        agent = None
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
            agent = PhoneAgent(model_config=model_config, agent_config=agent_config)

            # 发送开始事件
            event_queue.put(("event", {"type": "start", "message": "任务开始执行"}))

            # 第一步
            current_step_holder["step"] = 1
            step_start = time.time()
            step_result = agent.step(cmd)
            step_duration = time.time() - step_start

            while True:
                # 将 action 转为字符串
                action_str = ""
                if step_result.action:
                    if isinstance(step_result.action, dict):
                        action_str = json.dumps(step_result.action, ensure_ascii=False)
                    else:
                        action_str = str(step_result.action)

                # 检测 Take_over 动作
                is_takeover = 'Take_over' in action_str if action_str else False

                # 检测敏感操作
                sensitive_msg = None
                if action_str and not is_takeover:
                    is_sensitive_action = (
                        'Sensitive' in action_str or
                        'Confirm' in action_str or
                        '"action": "Sensitive"' in action_str or
                        'action="Sensitive"' in action_str
                    )
                    is_finish = 'finish' in action_str.lower() or '_metadata' in action_str

                    if is_sensitive_action and not is_finish:
                        msg_match = re.search(r'message["\s:=]+["\']?([^"\'}\]]+)', action_str)
                        if msg_match:
                            sensitive_msg = msg_match.group(1).strip()

                # 发送步骤完成事件
                event_queue.put(("event", {
                    "type": "step",
                    "step": agent.step_count,
                    "thinking": step_result.thinking,
                    "action": step_result.action,
                    "success": step_result.success,
                    "finished": step_result.finished,
                    "duration": round(step_duration, 3),
                    "takeover": is_takeover,
                    "sensitive": sensitive_msg is not None,
                    "sensitiveMessage": sensitive_msg,
                }))

                # 敏感操作处理
                if sensitive_msg:
                    event_queue.put(("event", {
                        "type": "sensitive",
                        "message": sensitive_msg,
                        "step": agent.step_count,
                        "action": step_result.action,
                    }))
                    event_queue.put(("event", {
                        "type": "done",
                        "message": f"等待确认敏感操作: {sensitive_msg}",
                        "steps": agent.step_count,
                        "success": True,
                        "paused": True,
                        "pauseReason": "sensitive",
                    }))
                    break

                # Take_over 处理
                if is_takeover:
                    # 尝试多种格式匹配 message
                    takeover_msg = None

                    # 当 AST 解析失败时，action 会被包装成 finish(message="原始action字符串")
                    # 所以需要先获取原始的 action 字符串
                    raw_action_str = action_str
                    if isinstance(step_result.action, dict) and step_result.action.get('_metadata') == 'finish':
                        # 从 finish 包装中提取原始 action 字符串
                        raw_action_str = step_result.action.get('message', '') or action_str

                    # 格式1: message="xxx" 或 message='xxx'（带引号，支持多行，使用 DOTALL）
                    # 使用贪婪匹配到最后一个引号，处理转义引号
                    match = re.search(r'message\s*=\s*"((?:[^"\\]|\\.)*)"', raw_action_str, re.DOTALL)
                    if not match:
                        match = re.search(r"message\s*=\s*'((?:[^'\\]|\\.)*)'", raw_action_str, re.DOTALL)
                    if match:
                        # 去掉转义字符，保留第一行作为简短消息
                        full_msg = match.group(1).replace('\\"', '"').replace("\\'", "'").strip()
                        # 处理字面量 \n（两个字符）和真实换行符
                        if '\\n' in full_msg:
                            takeover_msg = full_msg.split('\\n')[0]
                        elif '\n' in full_msg:
                            takeover_msg = full_msg.split('\n')[0]
                        else:
                            takeover_msg = full_msg

                    if not takeover_msg:
                        # 格式2: "message": "xxx"（JSON格式）
                        match = re.search(r'"message"\s*:\s*"((?:[^"\\]|\\.)*)"', action_str, re.DOTALL)
                        if match:
                            full_msg = match.group(1).replace('\\"', '"').strip()
                            if '\\n' in full_msg:
                                takeover_msg = full_msg.split('\\n')[0]
                            elif '\n' in full_msg:
                                takeover_msg = full_msg.split('\n')[0]
                            else:
                                takeover_msg = full_msg

                    # 如果还没找到，尝试从 step_result.action 对象中直接获取
                    if not takeover_msg and isinstance(step_result.action, dict):
                        full_msg = step_result.action.get('message', '')
                        if full_msg and 'Take_over' in full_msg:
                            # message 字段包含原始 action 字符串，再次解析
                            inner_match = re.search(r'message\s*=\s*"((?:[^"\\]|\\.)*)"', full_msg, re.DOTALL)
                            if inner_match:
                                full_msg = inner_match.group(1).replace('\\"', '"').strip()
                        if full_msg:
                            if '\\n' in full_msg:
                                takeover_msg = full_msg.split('\\n')[0]
                            elif '\n' in full_msg:
                                takeover_msg = full_msg.split('\n')[0]
                            else:
                                takeover_msg = full_msg

                    # 默认消息
                    if not takeover_msg:
                        takeover_msg = "需要手动操作，请完成后点击继续"
                    event_queue.put(("event", {
                        "type": "takeover",
                        "message": takeover_msg,
                        "step": agent.step_count,
                    }))
                    event_queue.put(("event", {
                        "type": "done",
                        "message": f"需要手动操作: {takeover_msg}",
                        "steps": agent.step_count,
                        "success": True,
                        "paused": True,
                        "pauseReason": "takeover",
                    }))
                    break

                if step_result.finished:
                    event_queue.put(("event", {
                        "type": "done",
                        "message": step_result.message,
                        "steps": agent.step_count,
                        "success": step_result.success,
                    }))
                    break

                if agent.step_count >= max_steps:
                    event_queue.put(("event", {
                        "type": "done",
                        "message": "已达到最大步数限制",
                        "steps": agent.step_count,
                        "success": False,
                    }))
                    break

                # 继续下一步
                current_step_holder["step"] = agent.step_count + 1
                step_start = time.time()
                step_result = agent.step()
                step_duration = time.time() - step_start

        except Exception as e:
            event_queue.put(("event", {"type": "error", "message": str(e)}))
        finally:
            if agent:
                agent.reset()
            unpatch_phone_agent(original_client)
            stop_event.set()

    async def event_generator():
        """异步 SSE 事件生成器"""
        loop = asyncio.get_event_loop()

        # 在后台线程中运行 agent
        agent_thread = threading.Thread(target=run_agent, daemon=True)
        agent_thread.start()

        while not stop_event.is_set() or not event_queue.empty():
            try:
                # 非阻塞获取事件
                msg = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: event_queue.get(timeout=0.05)),
                    timeout=0.1
                )
                msg_type, data = msg

                if msg_type == "token":
                    yield f"event: token\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
                else:
                    event_type = data.get("type", "message")
                    yield f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

                    # 如果是 done 或 error，结束生成
                    if event_type in ("done", "error"):
                        break
            except (queue.Empty, asyncio.TimeoutError):
                continue

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

