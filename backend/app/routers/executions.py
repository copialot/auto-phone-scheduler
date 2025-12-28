import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.database import get_db, async_session
from app.models.execution import Execution, ExecutionStatus
from app.models.task import Task
from app.schemas.execution import ExecutionResponse, ExecutionDetail
from app.services.execution_events import event_bus


def send_bytes_range(file_path: Path, start: int, end: int, chunk_size: int = 1024 * 1024):
    """生成文件的字节范围"""
    with open(file_path, "rb") as f:
        f.seek(start)
        remaining = end - start + 1
        while remaining > 0:
            chunk = f.read(min(chunk_size, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk

router = APIRouter(prefix="/api/executions", tags=["executions"])


@router.get("", response_model=list[ExecutionResponse])
async def list_executions(
    task_id: int | None = None,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    """获取执行历史列表"""
    query = select(Execution).options(joinedload(Execution.task))

    if task_id:
        query = query.where(Execution.task_id == task_id)

    query = query.order_by(Execution.started_at.desc()).offset(offset).limit(limit)

    result = await db.execute(query)
    executions = result.scalars().all()

    response = []
    for execution in executions:
        exec_response = ExecutionResponse(
            id=execution.id,
            task_id=execution.task_id,
            task_name=execution.task.name if execution.task else None,
            status=execution.status,
            started_at=execution.started_at,
            finished_at=execution.finished_at,
            error_message=execution.error_message,
        )
        response.append(exec_response)

    return response


@router.get("/count")
async def get_execution_count(
    task_id: int | None = None,
    db: AsyncSession = Depends(get_db),
):
    """获取执行记录总数"""
    from sqlalchemy import func

    query = select(func.count(Execution.id))
    if task_id:
        query = query.where(Execution.task_id == task_id)

    result = await db.execute(query)
    count = result.scalar()

    return {"count": count}


@router.get("/{execution_id}", response_model=ExecutionDetail)
async def get_execution(execution_id: int, db: AsyncSession = Depends(get_db)):
    """获取执行详情"""
    result = await db.execute(
        select(Execution)
        .options(joinedload(Execution.task))
        .where(Execution.id == execution_id)
    )
    execution = result.scalar_one_or_none()
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")

    return ExecutionDetail(
        id=execution.id,
        task_id=execution.task_id,
        task_name=execution.task.name if execution.task else None,
        status=execution.status,
        started_at=execution.started_at,
        finished_at=execution.finished_at,
        error_message=execution.error_message,
        steps=execution.steps,
        recording_path=execution.recording_path,
    )


@router.get("/{execution_id}/stream")
async def stream_execution(execution_id: int, db: AsyncSession = Depends(get_db)):
    """流式获取执行步骤（SSE）"""
    # 检查执行记录是否存在
    result = await db.execute(
        select(Execution)
        .options(joinedload(Execution.task))
        .where(Execution.id == execution_id)
    )
    execution = result.scalar_one_or_none()
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")

    async def event_generator():
        sent_steps = set()  # 记录已发送的步骤，避免重复

        # 如果执行已完成，发送所有步骤和完成事件
        if execution.status != ExecutionStatus.RUNNING:
            if execution.steps:
                for step in execution.steps:
                    yield f"event: step\ndata: {json.dumps(step, ensure_ascii=False)}\n\n"
            done_data = {
                "success": execution.status == ExecutionStatus.SUCCESS,
                "message": execution.error_message,
            }
            yield f"event: done\ndata: {json.dumps(done_data, ensure_ascii=False)}\n\n"
            return

        # 先订阅事件，再读取已有步骤，避免竞争条件
        q = await event_bus.subscribe(execution_id)
        try:
            # 订阅后重新读取数据库，获取最新的步骤
            async with async_session() as fresh_session:
                fresh_result = await fresh_session.execute(
                    select(Execution).where(Execution.id == execution_id)
                )
                fresh_execution = fresh_result.scalar_one_or_none()
                if fresh_execution and fresh_execution.steps:
                    for step in fresh_execution.steps:
                        step_num = step.get("step", 0)
                        if step_num not in sent_steps:
                            sent_steps.add(step_num)
                            yield f"event: step\ndata: {json.dumps(step, ensure_ascii=False)}\n\n"

                # 如果在读取期间任务已完成
                if fresh_execution and fresh_execution.status != ExecutionStatus.RUNNING:
                    done_data = {
                        "success": fresh_execution.status == ExecutionStatus.SUCCESS,
                        "message": fresh_execution.error_message,
                    }
                    yield f"event: done\ndata: {json.dumps(done_data, ensure_ascii=False)}\n\n"
                    return

            # 监听新事件
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=30.0)
                    event_type = event["type"]
                    event_data = event["data"]

                    # 避免重复发送步骤
                    if event_type == "step":
                        step_num = event_data.get("step", 0)
                        if step_num in sent_steps:
                            continue
                        sent_steps.add(step_num)

                    yield f"event: {event_type}\ndata: {json.dumps(event_data, ensure_ascii=False)}\n\n"

                    if event_type == "done":
                        break
                except asyncio.TimeoutError:
                    # 发送心跳保持连接
                    yield f"event: heartbeat\ndata: {{}}\n\n"
        finally:
            await event_bus.unsubscribe(execution_id, q)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/{execution_id}/recording")
async def get_recording(
    execution_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """获取录屏文件（支持 Range 请求以实现进度条拖动）"""
    result = await db.execute(
        select(Execution).where(Execution.id == execution_id)
    )
    execution = result.scalar_one_or_none()
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")

    if not execution.recording_path:
        raise HTTPException(status_code=404, detail="Recording not found")

    filepath = Path(execution.recording_path)
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Recording file not found")

    file_size = filepath.stat().st_size
    range_header = request.headers.get("range")

    # 如果没有 Range 请求头，返回完整文件
    if not range_header:
        return FileResponse(
            filepath,
            media_type="video/mp4",
            filename=filepath.name,
            headers={"Accept-Ranges": "bytes"},
        )

    # 解析 Range 请求头
    # 格式: bytes=start-end 或 bytes=start-
    try:
        range_spec = range_header.replace("bytes=", "")
        if "-" not in range_spec:
            raise ValueError("Invalid range format")

        parts = range_spec.split("-")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if parts[1] else file_size - 1

        # 限制范围
        start = max(0, start)
        end = min(end, file_size - 1)

        if start > end or start >= file_size:
            return Response(
                status_code=416,
                headers={"Content-Range": f"bytes */{file_size}"},
            )

        content_length = end - start + 1

        return StreamingResponse(
            send_bytes_range(filepath, start, end),
            status_code=206,
            media_type="video/mp4",
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(content_length),
            },
        )
    except (ValueError, IndexError):
        # Range 解析失败，返回完整文件
        return FileResponse(
            filepath,
            media_type="video/mp4",
            filename=filepath.name,
            headers={"Accept-Ranges": "bytes"},
        )


@router.delete("/{execution_id}")
async def delete_execution(execution_id: int, db: AsyncSession = Depends(get_db)):
    """删除执行记录"""
    result = await db.execute(
        select(Execution).where(Execution.id == execution_id)
    )
    execution = result.scalar_one_or_none()
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")

    # 删除录屏文件
    if execution.recording_path:
        filepath = Path(execution.recording_path)
        if filepath.exists():
            filepath.unlink()

    await db.delete(execution)
    await db.commit()

    return {"message": "Execution deleted"}


@router.delete("")
async def clear_all_executions(db: AsyncSession = Depends(get_db)):
    """清空所有执行记录"""
    result = await db.execute(select(Execution))
    executions = result.scalars().all()

    deleted_count = 0
    for execution in executions:
        # 删除录屏文件
        if execution.recording_path:
            filepath = Path(execution.recording_path)
            if filepath.exists():
                filepath.unlink()
        await db.delete(execution)
        deleted_count += 1

    await db.commit()

    return {"message": f"Deleted {deleted_count} executions"}
