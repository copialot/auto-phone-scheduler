/**
 * 聊天步骤展示组件
 * 在调试控制台和执行历史详情页面共享使用
 */
import { Bot, MousePointer, Play, Hand, CheckCircle, XCircle, AlertTriangle, Monitor } from 'lucide-react'

export interface StepInfo {
  step: number
  thinking: string
  action: Record<string, unknown> | string | null  // action 可能是对象或字符串
  finished: boolean
  success: boolean
  message: string | null
  duration?: number
}

export interface ChatItem {
  id: number
  type: 'user' | 'thinking' | 'action' | 'result' | 'error' | 'takeover' | 'sensitive'
  content: string
  timestamp: Date
  stepInfo?: StepInfo
  actionData?: Record<string, unknown>
}

/**
 * 解析 action 字符串为对象，支持多种格式
 * - JSON 对象: {"action": "Tap", "x": 100}
 * - 函数调用: Tap(x=100, y=200)
 * - 简单名称: Finish 或 [finish]
 */
export function parseActionToObject(action: unknown): Record<string, unknown> | null {
  if (!action) return null

  // 已经是对象
  if (typeof action === 'object') {
    return action as Record<string, unknown>
  }

  // 字符串格式
  if (typeof action === 'string') {
    const actionStr = action.trim()
    if (!actionStr) return null

    // 格式1: JSON 对象 {"action": "Tap", "x": 100}
    const jsonMatch = actionStr.match(/\{[\s\S]*?"action"[\s\S]*?\}/i)
    if (jsonMatch) {
      try {
        return JSON.parse(jsonMatch[0])
      } catch {
        // 继续尝试其他格式
      }
    }

    // 格式2: 函数调用格式 Tap(x=100, y=200) 或 Launch(package="com.xxx")
    const funcMatch = actionStr.match(/^(\w+)\((.*)\)$/s)
    if (funcMatch) {
      const actionName = funcMatch[1]
      const paramsStr = funcMatch[2]
      const result: Record<string, unknown> = { action: actionName }

      // 解析参数
      const paramRegex = /(\w+)=("([^"]*)"|'([^']*)'|(\d+)|(\w+))/g
      let match
      while ((match = paramRegex.exec(paramsStr)) !== null) {
        const key = match[1]
        const value = match[3] || match[4] || (match[5] ? parseInt(match[5]) : match[6])
        result[key] = value
      }

      return result
    }

    // 格式3: 简单的动作名称 如 Finish 或 [finish]
    const simpleMatch = actionStr.match(/^\[?(\w+)\]?$/i)
    if (simpleMatch) {
      return { action: simpleMatch[1] }
    }
  }

  return null
}

// 获取动作图标
function getActionIcon(actionName: string) {
  switch (actionName.toLowerCase()) {
    case 'tap': return <MousePointer className="h-3.5 w-3.5" />
    case 'launch': return <Play className="h-3.5 w-3.5" />
    case 'take_over': return <Hand className="h-3.5 w-3.5" />
    case 'systemlog': return <Monitor className="h-3.5 w-3.5" />
    default: return <Bot className="h-3.5 w-3.5" />
  }
}

// 根据动作类型选择颜色
function getActionColor(actionName: string) {
  switch (actionName.toLowerCase()) {
    case 'tap': return 'from-blue-500 to-cyan-500'
    case 'launch': return 'from-green-500 to-emerald-500'
    case 'scroll': return 'from-orange-500 to-amber-500'
    case 'input': return 'from-purple-500 to-pink-500'
    case 'take_over': return 'from-amber-500 to-yellow-500'
    case 'finish': return 'from-green-500 to-teal-500'
    case 'systemlog': return 'from-slate-500 to-zinc-500'
    default: return 'from-gray-500 to-slate-500'
  }
}

// 聊天项卡片组件
export function ChatItemCard({ item }: { item: ChatItem }) {
  // 用户消息
  if (item.type === 'user') {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] p-3 rounded-2xl rounded-br-sm bg-primary text-primary-foreground text-sm">
          {item.content}
        </div>
      </div>
    )
  }

  // AI 思考过程（流式显示）
  if (item.type === 'thinking') {
    // 检查是否是系统日志（step 为 0 且 action 是 SystemLog）
    const isSystemLog = item.stepInfo?.step === 0 &&
      typeof item.stepInfo?.action === 'object' &&
      item.stepInfo?.action !== null &&
      (item.stepInfo.action as Record<string, unknown>).action === 'SystemLog'

    return (
      <div className="flex gap-2">
        <div className={`flex-shrink-0 w-7 h-7 rounded-full flex items-center justify-center ${
          isSystemLog
            ? 'bg-gradient-to-br from-slate-500 to-zinc-600'
            : 'bg-gradient-to-br from-violet-500 to-purple-600'
        }`}>
          {isSystemLog ? (
            <Monitor className="h-4 w-4 text-white" />
          ) : (
            <Bot className="h-4 w-4 text-white" />
          )}
        </div>
        <div className="flex-1 max-w-[85%]">
          {!isSystemLog && (
            <div className="text-xs text-muted-foreground mb-1">
              Step {item.stepInfo?.step}
              {item.stepInfo?.duration !== undefined && (
                <span className="ml-2">{item.stepInfo.duration.toFixed(1)}s</span>
              )}
            </div>
          )}
          <div className={`p-3 rounded-2xl rounded-tl-sm text-sm whitespace-pre-wrap ${
            isSystemLog
              ? 'bg-slate-100 dark:bg-slate-900 text-slate-600 dark:text-slate-400'
              : 'bg-muted'
          }`}>
            {item.content || <span className="animate-pulse">●●●</span>}
          </div>
        </div>
      </div>
    )
  }

  // 动作卡片
  if (item.type === 'action' && item.actionData) {
    const action = item.actionData
    const actionName = (action.action as string) || (action._metadata as string) || 'action'
    const params = Object.entries(action).filter(([k]) => k !== '_metadata' && k !== 'action')

    return (
      <div className="flex gap-2 ml-9">
        <div className="flex-1 max-w-[85%]">
          <div className={`inline-flex items-center gap-2 px-3 py-2 rounded-xl bg-gradient-to-r ${getActionColor(actionName)} text-white text-sm font-medium shadow-sm`}>
            {getActionIcon(actionName)}
            <span>{actionName}</span>
            {params.length > 0 && (
              <span className="opacity-90 font-normal">
                {params.map(([k, v]) => (
                  <span key={k} className="ml-1">
                    {typeof v === 'string' ? v : JSON.stringify(v)}
                  </span>
                )).slice(0, 2)}
              </span>
            )}
          </div>
        </div>
      </div>
    )
  }

  // 执行结果
  if (item.type === 'result') {
    return (
      <div className="flex gap-2">
        <div className="flex-shrink-0 w-7 h-7 rounded-full bg-green-500 flex items-center justify-center">
          <CheckCircle className="h-4 w-4 text-white" />
        </div>
        <div className="flex-1 max-w-[85%] p-3 rounded-2xl rounded-tl-sm bg-green-50 dark:bg-green-950/50 border border-green-200 dark:border-green-800 text-sm">
          {item.content}
        </div>
      </div>
    )
  }

  // 错误消息
  if (item.type === 'error') {
    return (
      <div className="flex gap-2">
        <div className="flex-shrink-0 w-7 h-7 rounded-full bg-red-500 flex items-center justify-center">
          <XCircle className="h-4 w-4 text-white" />
        </div>
        <div className="flex-1 max-w-[85%] p-3 rounded-2xl rounded-tl-sm bg-red-50 dark:bg-red-950/50 border border-red-200 dark:border-red-800 text-sm text-red-700 dark:text-red-300">
          {item.content}
        </div>
      </div>
    )
  }

  // 需要手动接管
  if (item.type === 'takeover') {
    return (
      <div className="flex gap-2">
        <div className="flex-shrink-0 w-7 h-7 rounded-full bg-amber-500 flex items-center justify-center">
          <Hand className="h-4 w-4 text-white" />
        </div>
        <div className="flex-1 max-w-[85%] p-3 rounded-2xl rounded-tl-sm bg-amber-50 dark:bg-amber-950/50 border border-amber-200 dark:border-amber-800">
          <div className="text-sm font-medium text-amber-700 dark:text-amber-300 mb-1">需要手动操作</div>
          <div className="text-sm text-amber-600 dark:text-amber-400">{item.content}</div>
          <div className="text-xs text-amber-500 mt-2">完成后发送「继续」</div>
        </div>
      </div>
    )
  }

  // 敏感操作等待确认
  if (item.type === 'sensitive') {
    return (
      <div className="flex gap-2">
        <div className="flex-shrink-0 w-7 h-7 rounded-full bg-orange-500 flex items-center justify-center">
          <AlertTriangle className="h-4 w-4 text-white" />
        </div>
        <div className="flex-1 max-w-[85%] p-3 rounded-2xl rounded-tl-sm bg-orange-50 dark:bg-orange-950/50 border border-orange-200 dark:border-orange-800">
          <div className="text-sm font-medium text-orange-700 dark:text-orange-300 mb-1">敏感操作确认</div>
          <div className="text-sm text-orange-600 dark:text-orange-400">{item.content}</div>
          <div className="text-xs text-orange-500 mt-2">请在弹窗中确认或取消</div>
        </div>
      </div>
    )
  }

  return null
}
