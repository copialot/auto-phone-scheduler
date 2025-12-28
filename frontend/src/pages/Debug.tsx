import { useState, useRef, useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { Badge } from '@/components/ui/badge'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog'
import { ScrcpyPlayer } from '@/components/ScrcpyPlayer'
import { ChatItemCard, type ChatItem, type StepInfo, parseActionToObject } from '@/components/ChatSteps'
import { devicesApi, settingsApi, deviceConfigsApi } from '@/api/client'
import { Send, Smartphone, Loader2, AlertTriangle, Bot, Home, ArrowLeft, LayoutGrid, Hand, Unlock, Monitor, Lock, Power, Activity, XCircle } from 'lucide-react'
import { toast } from 'sonner'
import { Tooltip, TooltipContent, TooltipTrigger, TooltipProvider } from '@/components/ui/tooltip'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api'

// 敏感操作确认信息
interface SensitiveAction {
  message: string
  step: number
  action: string
}

export function Debug() {
  const [chatItems, setChatItems] = useState<ChatItem[]>([])
  const [input, setInput] = useState('')
  const [isExecuting, setIsExecuting] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const [streamKey, setStreamKey] = useState(0)
  const [useFallback, setUseFallback] = useState(false)
  // 触控模式
  const [touchEnabled, setTouchEnabled] = useState(true)
  // 敏感操作确认对话框状态
  const [sensitiveAction, setSensitiveAction] = useState<SensitiveAction | null>(null)
  const [pendingCommand, setPendingCommand] = useState<string | null>(null)
  // 流式 token 状态
  const [streamingThinking, setStreamingThinking] = useState('')
  const [streamingAction, setStreamingAction] = useState('')
  const currentStreamStepRef = useRef(0)

  const { data: devices = [] } = useQuery({
    queryKey: ['devices'],
    queryFn: devicesApi.list,
    refetchInterval: 5000,
  })

  const { data: settings } = useQuery({
    queryKey: ['settings'],
    queryFn: settingsApi.get,
  })

  const { data: deviceConfigs = [] } = useQuery({
    queryKey: ['device-configs'],
    queryFn: deviceConfigsApi.list,
  })

  // 获取当前设备（优先使用全局选定的设备，否则回退到第一个在线设备）
  const activeDevice = (() => {
    const onlineDevices = devices.filter(d => d.status === 'device')
    if (!onlineDevices.length) return undefined
    if (settings?.selected_device) {
      const selected = onlineDevices.find(d => d.serial === settings.selected_device)
      if (selected) return selected
    }
    return onlineDevices[0]
  })()

  // 获取当前设备的屏幕状态
  const { data: screenStatus, refetch: refetchScreenStatus } = useQuery({
    queryKey: ['screen-status', activeDevice?.serial],
    queryFn: () => activeDevice ? deviceConfigsApi.getScreenStatus(activeDevice.serial) : null,
    enabled: !!activeDevice,
    refetchInterval: 3000, // 每3秒刷新一次
  })

  // 获取当前设备的忙碌状态
  const { data: busyStatus, refetch: refetchBusyStatus } = useQuery({
    queryKey: ['device-busy-status', activeDevice?.serial],
    queryFn: () => activeDevice ? devicesApi.getBusyStatus(activeDevice.serial) : null,
    enabled: !!activeDevice,
    refetchInterval: 3000, // 每3秒刷新一次
  })

  // 滚动到底部
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [chatItems, streamingThinking, streamingAction])

  // 流式执行指令（使用 fetch + ReadableStream）
  const executeStream = async (command: string) => {
    const url = `${API_BASE}/debug/execute-stream`

    // 重置流式状态
    setStreamingThinking('')
    setStreamingAction('')
    currentStreamStepRef.current = 0

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command }),
      })

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.detail || '请求失败')
      }

      const reader = response.body?.getReader()
      if (!reader) throw new Error('无法读取响应流')

      const decoder = new TextDecoder()
      let buffer = ''
      let eventType = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          // 解析 SSE 事件类型
          if (line.startsWith('event: ')) {
            eventType = line.slice(7).trim()
            continue
          }

          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6))

              // 处理流式 token
              if (eventType === 'token' || data.type === 'token') {
                const { step, phase, content } = data
                if (step !== currentStreamStepRef.current) {
                  // 新步骤开始，清空流式内容
                  setStreamingThinking('')
                  setStreamingAction('')
                  currentStreamStepRef.current = step
                }
                if (phase === 'thinking') {
                  setStreamingThinking(prev => prev + content)
                } else if (phase === 'action') {
                  setStreamingAction(prev => prev + content)
                }
                continue
              }

              if (data.type === 'start') {
                console.log('[Debug] 任务开始执行')
              } else if (data.type === 'step') {
                const stepInfo = data as StepInfo
                const now = Date.now()
                // 检查是否是 takeover 或 sensitive 步骤
                const isTakeover = data.takeover === true
                const isSensitive = data.sensitive === true

                // 清空流式状态
                setStreamingThinking('')
                setStreamingAction('')

                // 添加思考消息（已完成的）
                if (stepInfo.thinking) {
                  setChatItems(prev => [...prev, {
                    id: now + stepInfo.step * 100,
                    type: 'thinking',
                    content: stepInfo.thinking,
                    timestamp: new Date(),
                    stepInfo,
                  }])
                }

                // 添加动作卡片（takeover 和 sensitive 由专门的事件处理，跳过这里）
                if (stepInfo.action && !isTakeover && !isSensitive) {
                  const actionData = parseActionToObject(stepInfo.action)
                  if (actionData) {
                    setChatItems(prev => [...prev, {
                      id: now + stepInfo.step * 100 + 1,
                      type: 'action',
                      content: '',
                      timestamp: new Date(),
                      stepInfo,
                      actionData,
                    }])
                  }
                }

                // 如果是最后一步，添加结果消息
                if (stepInfo.finished && stepInfo.message) {
                  setTimeout(() => {
                    setChatItems(prev => [...prev, {
                      id: now + 2000,
                      type: 'result',
                      content: stepInfo.message!,
                      timestamp: new Date(),
                    }])
                  }, 100)
                }
              } else if (data.type === 'sensitive') {
                // 敏感操作：需要用户确认
                setSensitiveAction({
                  message: data.message,
                  step: data.step,
                  action: data.action,
                })
                // 添加等待确认的消息
                setChatItems(prev => [...prev, {
                  id: Date.now(),
                  type: 'sensitive',
                  content: data.message,
                  timestamp: new Date(),
                }])
              } else if (data.type === 'takeover') {
                // Take_over 动作：需要用户手动操作
                setChatItems(prev => [...prev, {
                  id: Date.now(),
                  type: 'takeover',
                  content: data.message,
                  timestamp: new Date(),
                }])
              } else if (data.type === 'done') {
                // 清空流式状态
                setStreamingThinking('')
                setStreamingAction('')
                currentStreamStepRef.current = 0

                if (data.paused) {
                  // 暂停状态，等待用户手动操作后继续
                  setIsExecuting(false)
                  // 如果是敏感操作暂停，保存当前命令以便继续
                  if (data.pauseReason === 'sensitive') {
                    setPendingCommand(command)
                  }
                  return
                }
                if (data.message && !data.success) {
                  setChatItems(prev => [...prev, {
                    id: Date.now(),
                    type: 'result',
                    content: data.message,
                    timestamp: new Date(),
                  }])
                }
                setIsExecuting(false)
                return
              } else if (data.type === 'error') {
                // 清空流式状态
                setStreamingThinking('')
                setStreamingAction('')
                currentStreamStepRef.current = 0

                setChatItems(prev => [...prev, {
                  id: Date.now(),
                  type: 'error',
                  content: data.message,
                  timestamp: new Date(),
                }])
                setIsExecuting(false)
                return
              }
            } catch {
              // 忽略解析错误
            }
          }
        }
      }
      setIsExecuting(false)
    } catch (err) {
      setStreamingThinking('')
      setStreamingAction('')
      currentStreamStepRef.current = 0
      setChatItems(prev => [...prev, {
        id: Date.now(),
        type: 'error',
        content: err instanceof Error ? err.message : '连接错误',
        timestamp: new Date(),
      }])
      setIsExecuting(false)
    }
  }

  // 执行唤醒和解锁（如果设备有配置）
  const performWakeAndUnlock = async (): Promise<boolean> => {
    if (!activeDevice || !activeDeviceConfig) return true

    try {
      // 先检测屏幕状态
      let currentStatus = await deviceConfigsApi.getScreenStatus(activeDevice.serial)

      // 唤醒设备（如果屏幕未亮）
      if (activeDeviceConfig.wake_enabled && !currentStatus.screen_on) {
        setChatItems(prev => [...prev, {
          id: Date.now(),
          type: 'result',
          content: '正在唤醒设备...',
          timestamp: new Date(),
        }])
        const wakeResult = await deviceConfigsApi.testWake(activeDevice.serial, {
          wake_command: activeDeviceConfig.wake_command || undefined
        })
        if (!wakeResult.success) {
          setChatItems(prev => [...prev, {
            id: Date.now(),
            type: 'error',
            content: `唤醒失败: ${wakeResult.message}`,
            timestamp: new Date(),
          }])
          return false
        }
        // 等待设备唤醒
        await new Promise(resolve => setTimeout(resolve, 500))
        // 重新检测状态
        currentStatus = await deviceConfigsApi.getScreenStatus(activeDevice.serial)
      }

      // 解锁设备（如果处于锁屏状态）
      if (activeDeviceConfig.unlock_enabled && activeDeviceConfig.unlock_type && currentStatus.is_locked) {
        if (activeDeviceConfig.unlock_start_x === undefined || activeDeviceConfig.unlock_start_y === undefined) {
          setChatItems(prev => [...prev, {
            id: Date.now(),
            type: 'error',
            content: '设备未配置解锁坐标',
            timestamp: new Date(),
          }])
          return false
        }
        setChatItems(prev => [...prev, {
          id: Date.now(),
          type: 'result',
          content: '正在解锁设备...',
          timestamp: new Date(),
        }])

        // 最多重试 3 次
        const maxRetries = 3
        for (let attempt = 0; attempt < maxRetries; attempt++) {
          const unlockResult = await deviceConfigsApi.testUnlock(activeDevice.serial, {
            unlock_type: activeDeviceConfig.unlock_type,
            unlock_start_x: activeDeviceConfig.unlock_start_x!,
            unlock_start_y: activeDeviceConfig.unlock_start_y!,
            unlock_end_x: activeDeviceConfig.unlock_end_x ?? undefined,
            unlock_end_y: activeDeviceConfig.unlock_end_y ?? undefined,
            unlock_duration: activeDeviceConfig.unlock_duration ?? 300,
          })
          if (!unlockResult.success) {
            setChatItems(prev => [...prev, {
              id: Date.now(),
              type: 'error',
              content: `解锁失败: ${unlockResult.message}`,
              timestamp: new Date(),
            }])
            return false
          }

          // 等待解锁动画完成
          await new Promise(resolve => setTimeout(resolve, 500))

          // 检查是否已解锁
          const afterStatus = await deviceConfigsApi.getScreenStatus(activeDevice.serial)
          if (!afterStatus.is_locked) {
            // 解锁成功
            return true
          }

          // 仍然锁定，等待后重试
          if (attempt < maxRetries - 1) {
            setChatItems(prev => [...prev, {
              id: Date.now(),
              type: 'result',
              content: `解锁未成功，正在重试 (${attempt + 2}/${maxRetries})...`,
              timestamp: new Date(),
            }])
            await new Promise(resolve => setTimeout(resolve, 500))
          }
        }

        // 重试 3 次仍然锁定
        setChatItems(prev => [...prev, {
          id: Date.now(),
          type: 'error',
          content: `设备解锁失败：重试 ${maxRetries} 次后仍处于锁屏状态`,
          timestamp: new Date(),
        }])
        return false
      }

      return true
    } catch (err) {
      setChatItems(prev => [...prev, {
        id: Date.now(),
        type: 'error',
        content: `设备准备失败: ${err instanceof Error ? err.message : '未知错误'}`,
        timestamp: new Date(),
      }])
      return false
    }
  }

  const handleSend = async () => {
    if (!input.trim() || isExecuting) return

    const cmd = input.trim()

    // 添加用户消息
    setChatItems(prev => [...prev, {
      id: Date.now(),
      type: 'user',
      content: cmd,
      timestamp: new Date(),
    }])

    setIsExecuting(true)
    setInput('')

    // 先执行唤醒和解锁
    const prepared = await performWakeAndUnlock()
    if (!prepared) {
      setIsExecuting(false)
      return
    }

    executeStream(cmd)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  // 确认敏感操作后继续执行
  const handleConfirmSensitive = () => {
    setSensitiveAction(null)
    // 添加确认消息
    setChatItems(prev => [...prev, {
      id: Date.now(),
      type: 'result',
      content: '已确认操作，继续执行...',
      timestamp: new Date(),
    }])
    // 继续执行（使用 "继续" 命令让 agent 继续）
    if (pendingCommand) {
      setIsExecuting(true)
      executeStream('继续执行')
      setPendingCommand(null)
    }
  }

  // 取消敏感操作
  const handleCancelSensitive = () => {
    setSensitiveAction(null)
    setPendingCommand(null)
    // 添加取消消息
    setChatItems(prev => [...prev, {
      id: Date.now(),
      type: 'error',
      content: '已取消操作',
      timestamp: new Date(),
    }])
  }

  // 获取当前设备的配置
  const activeDeviceConfig = activeDevice
    ? deviceConfigs.find(c => c.device_serial === activeDevice.serial)
    : undefined

  // 设备控制操作
  const handleKeyEvent = async (key: 'home' | 'back' | 'app_switch') => {
    if (!activeDevice) return
    try {
      await devicesApi.sendKeyEvent(activeDevice.serial, key)
    } catch (err) {
      console.error(`发送 ${key} 按键失败:`, err)
    }
  }

  // 电源键（模拟实体电源键：息屏时点亮，亮屏时锁屏）
  const handlePowerKey = async () => {
    if (!activeDevice) return
    try {
      const status = await deviceConfigsApi.getScreenStatus(activeDevice.serial)
      if (status.screen_on) {
        // 屏幕亮着，锁屏
        const result = await deviceConfigsApi.lockScreen(activeDevice.serial)
        if (result.success) {
          toast.success('已锁定屏幕')
          refetchScreenStatus()
        } else {
          toast.error(result.message)
        }
      } else {
        // 屏幕关闭，唤醒
        const result = await deviceConfigsApi.testWake(activeDevice.serial, {
          wake_command: activeDeviceConfig?.wake_command || undefined
        })
        if (result.success) {
          toast.success('已唤醒屏幕')
          refetchScreenStatus()
        } else {
          toast.error(result.message)
        }
      }
    } catch (err) {
      toast.error('电源键操作失败')
      console.error('电源键操作失败:', err)
    }
  }

  // 释放设备（清除卡住的忙碌状态）
  const handleReleaseDevice = async () => {
    if (!activeDevice) return
    try {
      const result = await devicesApi.release(activeDevice.serial)
      if (result.success) {
        if (result.released_count > 0) {
          toast.success(result.message)
        } else {
          toast.info(result.message)
        }
        refetchBusyStatus()
      } else {
        toast.error('释放失败')
      }
    } catch (err) {
      toast.error('释放设备失败')
      console.error('释放设备失败:', err)
    }
  }

  // 解锁设备（使用已保存的配置）
  const handleUnlock = async () => {
    if (!activeDevice || !activeDeviceConfig) return
    if (!activeDeviceConfig.unlock_type) {
      toast.error('设备未配置解锁方式')
      return
    }
    if (activeDeviceConfig.unlock_start_x === undefined || activeDeviceConfig.unlock_start_y === undefined) {
      toast.error('设备未配置解锁坐标')
      return
    }
    try {
      // 先检查锁屏状态
      const status = await deviceConfigsApi.getScreenStatus(activeDevice.serial)
      if (!status.is_locked) {
        toast.info('设备未锁定，无需解锁')
        refetchScreenStatus()
        return
      }
      const result = await deviceConfigsApi.testUnlock(activeDevice.serial, {
        unlock_type: activeDeviceConfig.unlock_type,
        unlock_start_x: activeDeviceConfig.unlock_start_x!,
        unlock_start_y: activeDeviceConfig.unlock_start_y!,
        unlock_end_x: activeDeviceConfig.unlock_end_x ?? undefined,
        unlock_end_y: activeDeviceConfig.unlock_end_y ?? undefined,
        unlock_duration: activeDeviceConfig.unlock_duration ?? 300,
      })
      if (result.success) {
        toast.success(result.message)
        refetchScreenStatus()
      } else {
        toast.error(result.message)
      }
    } catch (err) {
      toast.error('解锁失败')
      console.error('解锁失败:', err)
    }
  }

  const handleSwipe = async (startX: number, startY: number, endX: number, endY: number) => {
    if (!activeDevice) return
    try {
      await devicesApi.sendSwipe(activeDevice.serial, startX, startY, endX, endY)
    } catch (err) {
      console.error('发送滑动失败:', err)
    }
  }

  const handleTap = async (x: number, y: number) => {
    if (!activeDevice) return
    try {
      // 如果屏幕关闭，先唤醒屏幕
      if (!screenStatus?.screen_on) {
        const result = await deviceConfigsApi.testWake(activeDevice.serial, {
          wake_command: activeDeviceConfig?.wake_command || undefined
        })
        if (result.success) {
          toast.success('已唤醒屏幕')
          refetchScreenStatus()
        } else {
          toast.error(result.message)
        }
        return
      }
      await devicesApi.sendTap(activeDevice.serial, x, y)
    } catch (err) {
      console.error('发送点击失败:', err)
    }
  }

  return (
    <TooltipProvider>
    <div className="h-full flex flex-col">
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-2xl font-bold">调试控制台</h1>
        <div className="flex items-center gap-2">
          {activeDevice ? (
            <>
              <Badge variant="success" className="flex items-center gap-1">
                <Smartphone className="h-3 w-3" />
                {activeDevice.model || activeDevice.serial}
              </Badge>
              {/* 屏幕状态显示 */}
              {screenStatus && (
                <>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Badge variant={screenStatus.screen_on ? 'secondary' : 'default'} className="flex items-center gap-1">
                        <Monitor className="h-3 w-3" />
                        {screenStatus.screen_on ? '亮屏' : '息屏'}
                      </Badge>
                    </TooltipTrigger>
                    <TooltipContent>{screenStatus.screen_on ? '屏幕已亮起' : '屏幕已关闭'}</TooltipContent>
                  </Tooltip>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Badge variant={screenStatus.is_locked ? 'destructive' : 'secondary'} className="flex items-center gap-1">
                        <Lock className="h-3 w-3" />
                        {screenStatus.is_locked ? '已锁定' : '未锁定'}
                      </Badge>
                    </TooltipTrigger>
                    <TooltipContent>{screenStatus.is_locked ? '设备处于锁屏状态' : '设备已解锁'}</TooltipContent>
                  </Tooltip>
                </>
              )}
              {/* 设备忙碌状态显示 */}
              {busyStatus && (
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Badge
                      variant={busyStatus.is_busy ? 'destructive' : 'secondary'}
                      className={`flex items-center gap-1 ${busyStatus.is_busy ? 'cursor-pointer hover:bg-destructive/80' : ''}`}
                      onClick={busyStatus.is_busy ? handleReleaseDevice : undefined}
                    >
                      <Activity className="h-3 w-3" />
                      {busyStatus.is_busy ? '忙碌' : '空闲'}
                      {busyStatus.is_busy && <XCircle className="h-3 w-3 ml-0.5" />}
                    </Badge>
                  </TooltipTrigger>
                  <TooltipContent>
                    {busyStatus.is_busy
                      ? `正在执行: ${busyStatus.task_name || `执行记录 #${busyStatus.execution_id}`}（点击释放）`
                      : '设备空闲，可以执行任务'
                    }
                  </TooltipContent>
                </Tooltip>
              )}
            </>
          ) : (
            <Badge variant="destructive">未连接设备</Badge>
          )}
        </div>
      </div>

      <div className="flex-1 grid grid-cols-1 lg:grid-cols-2 gap-4 min-h-0 overflow-hidden">
        {/* 左侧 - 对话区域 */}
        <Card className="flex flex-col min-h-0 h-full overflow-hidden">
          <CardHeader className="pb-3 shrink-0 border-b">
            <CardTitle className="text-base">执行步骤</CardTitle>
          </CardHeader>
          <CardContent className="flex-1 overflow-y-auto p-4 space-y-3 min-h-0">
            {chatItems.length === 0 && !isExecuting ? (
              <div className="text-center text-muted-foreground py-8">
                发送指令开始调试
              </div>
            ) : (
              chatItems.map(item => (
                <ChatItemCard key={item.id} item={item} />
              ))
            )}
            {/* 流式输出显示 - 与 ChatItemCard thinking 样式一致 */}
            {isExecuting && (streamingThinking || streamingAction) && (
              <div className="flex gap-2">
                <div className="flex-shrink-0 w-7 h-7 rounded-full bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center">
                  <Bot className="h-4 w-4 text-white" />
                </div>
                <div className="flex-1 max-w-[85%]">
                  <div className="text-xs text-muted-foreground mb-1">
                    Step {currentStreamStepRef.current || 1}
                  </div>
                  <div className="p-3 rounded-2xl rounded-tl-sm bg-muted text-sm whitespace-pre-wrap">
                    {streamingThinking}
                    {streamingAction && (
                      <span className="text-muted-foreground">{streamingAction}</span>
                    )}
                    <span className="inline-block w-1.5 h-4 bg-foreground/60 ml-0.5 animate-pulse align-middle" />
                  </div>
                </div>
              </div>
            )}
            {/* 等待状态（无流式内容时）- 跳动的省略号 */}
            {isExecuting && !streamingThinking && !streamingAction && (
              <div className="flex gap-2">
                <div className="flex-shrink-0 w-7 h-7 rounded-full bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center">
                  <Bot className="h-4 w-4 text-white" />
                </div>
                <div className="flex-1 max-w-[85%]">
                  <div className="text-xs text-muted-foreground mb-1">
                    Step {currentStreamStepRef.current || 1}
                  </div>
                  <div className="p-3 rounded-2xl rounded-tl-sm bg-muted text-sm flex items-center gap-0.5">
                    <span className="inline-block w-1 h-1 rounded-full bg-foreground/50 animate-bounce" style={{ animationDelay: '0ms' }} />
                    <span className="inline-block w-1 h-1 rounded-full bg-foreground/50 animate-bounce" style={{ animationDelay: '150ms' }} />
                    <span className="inline-block w-1 h-1 rounded-full bg-foreground/50 animate-bounce" style={{ animationDelay: '300ms' }} />
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </CardContent>
          {/* 输入区域 */}
          <div className="p-4 border-t shrink-0">
            <div className="flex gap-2">
              <Textarea
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="输入指令，如：打开微信..."
                className="min-h-15 max-h-30 resize-none"
                disabled={isExecuting || !activeDevice}
              />
              <Button
                onClick={handleSend}
                disabled={!input.trim() || isExecuting || !activeDevice}
                className="shrink-0"
              >
                {isExecuting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
              </Button>
            </div>
          </div>
        </Card>

        {/* 右侧 - 实时画面 */}
        <Card className="flex flex-col min-h-0 h-full overflow-hidden">
          <CardHeader className="pb-3 shrink-0 border-b">
            <div className="flex items-center justify-between">
              <CardTitle className="text-base flex items-center gap-2">
                <Smartphone className="h-4 w-4" />
                实时画面
              </CardTitle>
              <div className="flex items-center gap-1">
                {activeDevice && (
                  <>
                    {/* 触控模式开关 */}
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Button
                          variant={touchEnabled ? 'default' : 'ghost'}
                          size="sm"
                          onClick={() => setTouchEnabled(!touchEnabled)}
                        >
                          <Hand className="h-4 w-4" />
                        </Button>
                      </TooltipTrigger>
                      <TooltipContent>
                        {touchEnabled ? '触控已启用（点击禁用）' : '触控已禁用（点击启用）'}
                      </TooltipContent>
                    </Tooltip>
                    {/* 电源键（亮屏时锁屏，息屏时唤醒） */}
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Button variant="ghost" size="sm" onClick={handlePowerKey}>
                          <Power className="h-4 w-4" />
                        </Button>
                      </TooltipTrigger>
                      <TooltipContent>
                        {screenStatus?.screen_on ? '锁定屏幕' : '唤醒屏幕'}
                      </TooltipContent>
                    </Tooltip>
                    {/* 解锁按钮（如果设备有配置、屏幕亮起且处于锁屏状态） */}
                    {activeDeviceConfig?.unlock_enabled && screenStatus?.screen_on && screenStatus?.is_locked && (
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Button variant="ghost" size="sm" onClick={handleUnlock}>
                            <Unlock className="h-4 w-4" />
                          </Button>
                        </TooltipTrigger>
                        <TooltipContent>解锁屏幕</TooltipContent>
                      </Tooltip>
                    )}
                    {/* 导航按钮 */}
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Button variant="ghost" size="sm" onClick={() => handleKeyEvent('back')}>
                          <ArrowLeft className="h-4 w-4" />
                        </Button>
                      </TooltipTrigger>
                      <TooltipContent>返回</TooltipContent>
                    </Tooltip>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Button variant="ghost" size="sm" onClick={() => handleKeyEvent('home')}>
                          <Home className="h-4 w-4" />
                        </Button>
                      </TooltipTrigger>
                      <TooltipContent>主页</TooltipContent>
                    </Tooltip>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Button variant="ghost" size="sm" onClick={() => handleKeyEvent('app_switch')}>
                          <LayoutGrid className="h-4 w-4" />
                        </Button>
                      </TooltipTrigger>
                      <TooltipContent>最近任务</TooltipContent>
                    </Tooltip>
                    <Button variant="ghost" size="sm" onClick={() => setStreamKey(k => k + 1)}>
                      刷新
                    </Button>
                  </>
                )}
              </div>
            </div>
          </CardHeader>
          <CardContent className="flex-1 flex items-center justify-center p-4 bg-muted/30 overflow-hidden">
            {activeDevice ? (
              useFallback ? (
                <img
                  key={streamKey}
                  src={devicesApi.getStreamUrl(activeDevice.serial)}
                  alt="设备屏幕"
                  className="max-h-full w-auto rounded-lg shadow-lg"
                  style={{ maxWidth: '280px', objectFit: 'contain' }}
                />
              ) : (
                <ScrcpyPlayer
                  key={streamKey}
                  deviceId={activeDevice.serial}
                  className="max-h-full"
                  onFallback={() => setUseFallback(true)}
                  fallbackTimeout={8000}
                  enableTouch={touchEnabled}
                  onSwipe={handleSwipe}
                  onTap={handleTap}
                />
              )
            ) : (
              <div className="text-center text-muted-foreground">
                <Smartphone className="h-12 w-12 mx-auto mb-2 opacity-30" />
                <p>未连接设备</p>
                <p className="text-xs mt-1">请通过 ADB 连接手机</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* 敏感操作确认对话框 */}
      <AlertDialog open={sensitiveAction !== null} onOpenChange={(open) => !open && handleCancelSensitive()}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle className="flex items-center gap-2">
              <AlertTriangle className="h-5 w-5 text-amber-500" />
              敏感操作确认
            </AlertDialogTitle>
            <AlertDialogDescription className="space-y-2">
              <p>AI 正在尝试执行以下敏感操作：</p>
              <p className="font-medium text-foreground bg-muted p-3 rounded-lg">
                {sensitiveAction?.message}
              </p>
              <p className="text-sm">请确认是否允许执行此操作？</p>
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel onClick={handleCancelSensitive}>取消</AlertDialogCancel>
            <AlertDialogAction onClick={handleConfirmSensitive} className="bg-amber-600 hover:bg-amber-700">
              确认执行
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
    </TooltipProvider>
  )
}
