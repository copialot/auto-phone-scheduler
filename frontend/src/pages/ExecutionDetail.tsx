import { useEffect, useRef, useMemo, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { ChatItemCard, type ChatItem, parseActionToObject } from '@/components/ChatSteps'
import { ScrcpyPlayer } from '@/components/ScrcpyPlayer'
import { ScreenshotViewer } from '@/components/ScreenshotViewer'
import { executionsApi, devicesApi } from '@/api/client'
import { formatDateTime } from '@/lib/utils'
import { ArrowLeft, Video, MessageSquare, Clock, AlertCircle, Smartphone, Bot, Download } from 'lucide-react'
import type { ExecutionStep, Device } from '@/types'

// 流式 token 事件类型
interface TokenEvent {
  step: number
  phase: 'thinking' | 'action'
  content: string
}

// 解析步骤数据，支持新旧两种格式
function parseStepData(step: ExecutionStep): { thinking: string | null; action: Record<string, unknown> | null } {
  let thinking: string | null = null
  let action: Record<string, unknown> | null = null

  // 新格式：直接从字段读取 thinking
  if (step.thinking) {
    thinking = step.thinking
  }

  // 解析 action - 可能是对象或字符串
  if (step.action) {
    action = parseActionToObject(step.action)
  }

  // 如果新格式字段有值，直接返回
  if (thinking || action) {
    return { thinking, action }
  }

  // 旧格式：从 description 解析
  const description = step.description || ''
  if (!description) {
    return { thinking: null, action: null }
  }

  const thinkMatch = description.match(/<think(?:ing)?>([\s\S]*?)<\/think(?:ing)?>/i)
  const answerMatch = description.match(/<answer>([\s\S]*?)<\/answer>/i)

  if (thinkMatch) {
    thinking = thinkMatch[1].trim()
  }

  let answer: string = description
  if (answerMatch) {
    answer = answerMatch[1].trim()
  } else {
    answer = description
      .replace(/<think(?:ing)?>[\s\S]*?<\/think(?:ing)?>/gi, '')
      .replace(/<\/?answer>/gi, '')
      .trim()
  }

  // 尝试解析动作（使用通用解析函数）
  action = parseActionToObject(answer)

  return { thinking, action }
}

// 将 ExecutionStep 转换为 ChatItem 列表
function convertStepsToChatItems(steps: ExecutionStep[]): ChatItem[] {
  const items: ChatItem[] = []
  let id = 1

  for (const step of steps) {
    const { thinking, action } = parseStepData(step)
    const timestamp = new Date(step.timestamp)

    // 添加思考消息
    if (thinking) {
      items.push({
        id: id++,
        type: 'thinking',
        content: thinking,
        timestamp,
        stepInfo: {
          step: step.step,
          thinking,
          action,
          finished: false,
          success: true,
          message: null,
        },
      })
    }

    // 添加动作卡片（SystemLog 类型不显示动作卡片，只显示消息）
    if (action && (action.action as string)?.toLowerCase() !== 'systemlog') {
      items.push({
        id: id++,
        type: 'action',
        content: '',
        timestamp,
        actionData: action,
        stepInfo: {
          step: step.step,
          thinking: thinking || '',
          action,
          finished: false,
          success: true,
          message: null,
        },
      })
    } else if (!thinking && step.description) {
      // 没有思考和动作，显示描述内容
      const cleanDesc = step.description
        .replace(/<think(?:ing)?>[\s\S]*?<\/think(?:ing)?>/gi, '')
        .replace(/<\/?answer>/gi, '')
        .trim()
      if (cleanDesc) {
        items.push({
          id: id++,
          type: 'thinking',
          content: cleanDesc,
          timestamp,
          stepInfo: {
            step: step.step,
            thinking: cleanDesc,
            action: null,
            finished: false,
            success: true,
            message: null,
          },
        })
      }
    }
  }

  return items
}

export function ExecutionDetail() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const stepsEndRef = useRef<HTMLDivElement>(null)
  const [useFallback, setUseFallback] = useState(false)
  // 保存首次获取到的设备ID，避免后续刷新导致组件重新挂载
  const [stableDeviceId, setStableDeviceId] = useState<string | null>(null)
  // SSE 流式步骤
  const [streamSteps, setStreamSteps] = useState<ExecutionStep[]>([])
  const [isStreaming, setIsStreaming] = useState(false)
  const eventSourceRef = useRef<EventSource | null>(null)
  // 流式打字机效果状态
  const [streamingThinking, setStreamingThinking] = useState<string>('')
  const [streamingAction, setStreamingAction] = useState<string>('')
  const [currentStreamStep, setCurrentStreamStep] = useState<number>(0)

  // 初次获取执行记录（只获取一次基本信息）
  const { data: execution, isLoading, refetch } = useQuery({
    queryKey: ['execution', id],
    queryFn: () => executionsApi.get(Number(id)),
    enabled: !!id,
    staleTime: Infinity, // 不自动刷新，由 SSE 更新
  })

  // SSE 流式获取步骤
  useEffect(() => {
    if (!id || !execution) return

    // 如果执行已完成，不需要订阅 SSE
    if (execution.status !== 'running') {
      setStreamSteps(execution.steps || [])
      return
    }

    // 开始流式获取
    setIsStreaming(true)
    setStreamingThinking('')
    setStreamingAction('')
    setCurrentStreamStep(0)

    const eventSource = new EventSource(executionsApi.getStreamUrl(Number(id)))
    eventSourceRef.current = eventSource

    // 使用 ref 来跟踪当前步骤，避免闭包问题
    let currentStep = 0

    // 处理流式 token 事件（打字机效果）
    eventSource.addEventListener('token', (event) => {
      const token = JSON.parse(event.data) as TokenEvent
      // 如果是新步骤，重置流式状态
      if (token.step !== currentStep) {
        currentStep = token.step
        setCurrentStreamStep(token.step)
        setStreamingThinking('')
        setStreamingAction('')
      }
      // 追加 token
      if (token.phase === 'thinking') {
        setStreamingThinking(prev => prev + token.content)
      } else if (token.phase === 'action') {
        setStreamingAction(prev => prev + token.content)
      }
    })

    // 处理完整步骤事件
    eventSource.addEventListener('step', (event) => {
      const step = JSON.parse(event.data) as ExecutionStep
      setStreamSteps(prev => {
        // 避免重复添加：对于 SystemLog (step=0) 使用 timestamp 判重，其他用 step 判重
        if (step.step === 0) {
          // SystemLog 使用 timestamp 判重
          if (prev.some(s => s.step === 0 && s.timestamp === step.timestamp)) return prev
        } else {
          // 普通步骤使用 step 判重
          if (prev.some(s => s.step === step.step)) return prev
        }
        return [...prev, step]
      })
      // 清除该步骤的流式状态
      setStreamingThinking('')
      setStreamingAction('')
    })

    eventSource.addEventListener('done', async () => {
      setStreamingThinking('')
      setStreamingAction('')
      eventSource.close()
      // 强制刷新执行记录获取最终状态，等待完成后再关闭 streaming 状态
      await refetch()
      setIsStreaming(false)
    })

    eventSource.addEventListener('error', () => {
      setIsStreaming(false)
      eventSource.close()
    })

    return () => {
      eventSource.close()
      eventSourceRef.current = null
    }
  }, [id, execution?.status, refetch])  // 移除 currentStreamStep 依赖

  // 获取设备列表
  const { data: devices = [] } = useQuery({
    queryKey: ['devices'],
    queryFn: devicesApi.list,
    refetchInterval: 5000,
  })

  // 使用执行记录中的设备 serial 设置稳定的设备ID
  useMemo(() => {
    // 优先使用执行记录中保存的设备 serial
    if (execution?.device_serial && stableDeviceId === null) {
      // 验证设备是否在线
      const onlineDevices = devices.filter((d: Device) => d.status === 'device')
      const targetDevice = onlineDevices.find(d => d.serial === execution.device_serial)
      if (targetDevice) {
        setTimeout(() => setStableDeviceId(execution.device_serial!), 0)
        return
      }
    }

    // 如果执行记录没有设备信息，回退到第一个在线设备
    if (stableDeviceId === null) {
      const onlineDevices = devices.filter((d: Device) => d.status === 'device')
      if (onlineDevices.length > 0) {
        setTimeout(() => setStableDeviceId(onlineDevices[0].serial), 0)
      }
    }
  }, [devices, execution?.device_serial, stableDeviceId])

  // 使用流式步骤或执行记录中的步骤
  const steps = useMemo(() => {
    return isStreaming || execution?.status === 'running' ? streamSteps : (execution?.steps || [])
  }, [isStreaming, execution?.status, execution?.steps, streamSteps])

  // 将步骤转换为聊天项
  const chatItems = useMemo(() => {
    if (!steps.length) return []
    return convertStepsToChatItems(steps)
  }, [steps])

  // 自动滚动到最新步骤
  useEffect(() => {
    if (isStreaming || execution?.status === 'running') {
      stepsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }
  }, [chatItems, isStreaming, execution?.status])

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'success':
        return <Badge variant="success">成功</Badge>
      case 'failed':
        return <Badge variant="destructive">失败</Badge>
      case 'running':
        return <Badge variant="warning">运行中</Badge>
      default:
        return <Badge variant="secondary">等待</Badge>
    }
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-muted-foreground">加载中...</div>
      </div>
    )
  }

  if (!execution) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-muted-foreground">执行记录不存在</div>
      </div>
    )
  }

  const isRunning = execution.status === 'running' || isStreaming

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center gap-4 mb-4">
        <Button variant="ghost" size="sm" onClick={() => navigate('/tasks')}>
          <ArrowLeft className="h-4 w-4 mr-2" />
          返回
        </Button>
        <h1 className="text-xl font-bold">{execution.task_name}</h1>
        {getStatusBadge(execution.status)}
        {isRunning && (
          <span className="text-sm text-muted-foreground animate-pulse">
            执行中...
          </span>
        )}
      </div>

      {/* Info Bar */}
      <div className="flex items-center gap-6 text-sm text-muted-foreground mb-4">
        <div className="flex items-center gap-1">
          <Clock className="h-4 w-4" />
          <span>
            开始: {formatDateTime(execution.started_at)}
          </span>
        </div>
        {execution.finished_at && (
          <div className="flex items-center gap-1">
            <Clock className="h-4 w-4" />
            <span>结束: {formatDateTime(execution.finished_at)}</span>
          </div>
        )}
      </div>

      {/* Error Message */}
      {execution.error_message && (
        <div className="p-3 rounded-lg bg-destructive/10 text-destructive mb-4 flex items-start gap-2">
          <AlertCircle className="h-4 w-4 mt-0.5 shrink-0" />
          <div>
            <span className="font-medium">错误信息:</span>
            <p className="mt-1 text-sm whitespace-pre-wrap">{execution.error_message}</p>
          </div>
        </div>
      )}

      {/* Main Content - Two Column Layout with Fixed Height */}
      <div className="flex-1 grid grid-cols-1 lg:grid-cols-2 gap-4 min-h-0 overflow-hidden">
        {/* Left Column - Chat Style Dialog History (复用调试控制台UI) */}
        <Card className="flex flex-col min-h-0 h-full overflow-hidden">
          <CardHeader className="pb-3 shrink-0 border-b">
            <CardTitle className="text-base flex items-center gap-2">
              <MessageSquare className="h-4 w-4" />
              执行步骤 {chatItems.length > 0 ? `(${chatItems.length/2})` : ''}
            </CardTitle>
          </CardHeader>
          <CardContent className="flex-1 overflow-y-auto min-h-0 p-4 space-y-3">
            {chatItems.length > 0 || streamingThinking || streamingAction ? (
              <>
                {chatItems.map(item => (
                  <ChatItemCard key={item.id} item={item} />
                ))}
                {/* 流式打字机效果：显示正在输出的 thinking（复用 ChatItemCard 样式） */}
                {streamingThinking && (
                  <div className="flex gap-2">
                    <div className="shrink-0 w-7 h-7 rounded-full bg-linear-to-br from-violet-500 to-purple-600 flex items-center justify-center">
                      <Bot className="h-4 w-4 text-white" />
                    </div>
                    <div className="flex-1 max-w-[85%]">
                      <div className="text-xs text-muted-foreground mb-1">
                        Step {currentStreamStep}
                      </div>
                      <div className="p-3 rounded-2xl rounded-tl-sm bg-muted text-sm whitespace-pre-wrap">
                        {streamingThinking}
                        <span className="inline-block w-0.5 h-4 bg-violet-500 animate-pulse ml-0.5 align-middle" />
                      </div>
                    </div>
                  </div>
                )}
                {/* 流式打字机效果：显示正在输出的 action（复用 ChatItemCard 样式） */}
                {streamingAction && !streamingThinking && (
                  <div className="flex gap-2 ml-9">
                    <div className="flex-1 max-w-[85%]">
                      <div className="inline-flex items-center gap-2 px-3 py-2 rounded-xl bg-linear-to-r from-blue-500 to-cyan-500 text-white text-sm font-medium shadow-sm">
                        <Bot className="h-3.5 w-3.5" />
                        <span className="font-mono">{streamingAction}</span>
                        <span className="inline-block w-0.5 h-4 bg-white/70 animate-pulse" />
                      </div>
                    </div>
                  </div>
                )}
                {/* 等待 AI 输出：跳动的省略号 */}
                {isRunning && !streamingThinking && !streamingAction && (
                  <div className="flex gap-2">
                    <div className="shrink-0 w-7 h-7 rounded-full bg-linear-to-br from-violet-500 to-purple-600 flex items-center justify-center">
                      <Bot className="h-4 w-4 text-white" />
                    </div>
                    <div className="flex-1">
                      <div className="text-xs text-muted-foreground mb-1">
                        Step {currentStreamStep + 1}
                      </div>
                      <div className="p-3 rounded-2xl rounded-tl-sm bg-muted text-sm flex items-center gap-0.5">
                        <span className="inline-block w-1 h-1 rounded-full bg-violet-500 animate-bounce" style={{ animationDelay: '0ms' }} />
                        <span className="inline-block w-1 h-1 rounded-full bg-violet-500 animate-bounce" style={{ animationDelay: '150ms' }} />
                        <span className="inline-block w-1 h-1 rounded-full bg-violet-500 animate-bounce" style={{ animationDelay: '300ms' }} />
                      </div>
                    </div>
                  </div>
                )}
                <div ref={stepsEndRef} />
              </>
            ) : isRunning ? (
              <div className="flex gap-2">
                <div className="shrink-0 w-7 h-7 rounded-full bg-linear-to-br from-violet-500 to-purple-600 flex items-center justify-center">
                  <Bot className="h-4 w-4 text-white" />
                </div>
                <div className="flex-1">
                  <div className="text-xs text-muted-foreground mb-1">Step 1</div>
                  <div className="p-3 rounded-2xl rounded-tl-sm bg-muted text-sm flex items-center gap-0.5">
                    <span className="inline-block w-1 h-1 rounded-full bg-violet-500 animate-bounce" style={{ animationDelay: '0ms' }} />
                    <span className="inline-block w-1 h-1 rounded-full bg-violet-500 animate-bounce" style={{ animationDelay: '150ms' }} />
                    <span className="inline-block w-1 h-1 rounded-full bg-violet-500 animate-bounce" style={{ animationDelay: '300ms' }} />
                  </div>
                </div>
              </div>
            ) : (
              <div className="flex items-center justify-center h-full text-muted-foreground">
                暂无执行步骤
              </div>
            )}
          </CardContent>
        </Card>

        {/* Right Column - Live Stream or Recording */}
        <Card className="flex flex-col min-h-0 h-full overflow-hidden">
          <CardHeader className="pb-3 shrink-0 border-b">
            <div className="flex items-center justify-between">
              <CardTitle className="text-base flex items-center gap-2">
                <Video className="h-4 w-4" />
                {isRunning ? '实时屏幕' : '屏幕录像'}
                {isRunning && (
                  <span className="flex h-2 w-2">
                    <span className="animate-ping absolute inline-flex h-2 w-2 rounded-full bg-red-400 opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-red-500"></span>
                  </span>
                )}
              </CardTitle>
              {!isRunning && execution.recording_path && (
                <Button
                  variant="ghost"
                  size="sm"
                  asChild
                >
                  <a
                    href={executionsApi.getRecordingUrl(execution.id)}
                    download={`recording-${execution.id}.mp4`}
                  >
                    <Download className="h-4 w-4 mr-1" />
                    下载
                  </a>
                </Button>
              )}
            </div>
          </CardHeader>
          <CardContent className="flex-1 flex items-center justify-center p-4 bg-muted/30 overflow-hidden">
            {isRunning ? (
              // 运行中：显示实时屏幕
              stableDeviceId ? (
                useFallback ? (
                  <ScreenshotViewer
                    deviceId={stableDeviceId}
                    className="max-h-full"
                    refreshInterval={500}
                  />
                ) : (
                  <ScrcpyPlayer
                    deviceId={stableDeviceId}
                    className="max-h-full"
                    onFallback={() => setUseFallback(true)}
                    fallbackTimeout={15000}
                  />
                )
              ) : (
                <div className="text-center text-muted-foreground">
                  <Smartphone className="h-12 w-12 mx-auto mb-2 opacity-30" />
                  <p>等待设备连接...</p>
                </div>
              )
            ) : execution.recording_path ? (
              // 已完成：显示录像
              <video
                controls
                autoPlay
                className="w-full max-h-full rounded-lg object-contain"
                src={executionsApi.getRecordingUrl(execution.id)}
              >
                您的浏览器不支持视频播放
              </video>
            ) : (
              <div className="text-center text-muted-foreground">
                <Video className="h-12 w-12 mx-auto mb-2 opacity-30" />
                <p>无录屏文件</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
