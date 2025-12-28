export interface Task {
  id: number
  name: string
  description: string | null
  command: string
  cron_expression: string
  timezone: string | null  // IANA 时区名称
  enabled: boolean
  notify_on_success: boolean
  notify_on_failure: boolean
  notification_channel_ids: number[] | null  // 空或 null 表示使用所有启用的渠道
  auto_confirm_sensitive: boolean  // 敏感操作自动确认
  device_serial: string | null  // 指定执行设备，为空则使用全局设置
  random_delay_minutes: number | null  // 随机延迟时间区间（分钟）
  wake_before_run: boolean  // 执行前唤醒设备
  unlock_before_run: boolean  // 执行前解锁设备
  go_home_after_run: boolean  // 执行完成后返回主屏幕
  created_at: string
  updated_at: string
  next_run: string | null
}

export interface TaskCreate {
  name: string
  description?: string
  command: string
  cron_expression: string
  timezone?: string  // IANA 时区名称
  enabled?: boolean
  notify_on_success?: boolean
  notify_on_failure?: boolean
  notification_channel_ids?: number[] | null
  auto_confirm_sensitive?: boolean  // 敏感操作自动确认
  device_serial?: string | null  // 指定执行设备
  random_delay_minutes?: number | null  // 随机延迟时间区间（分钟）
  wake_before_run?: boolean  // 执行前唤醒设备
  unlock_before_run?: boolean  // 执行前解锁设备
  go_home_after_run?: boolean  // 执行完成后返回主屏幕
}

export interface TaskUpdate {
  name?: string
  description?: string
  command?: string
  cron_expression?: string
  timezone?: string
  enabled?: boolean
  notify_on_success?: boolean
  notify_on_failure?: boolean
  notification_channel_ids?: number[] | null
  device_serial?: string | null
  random_delay_minutes?: number | null
  wake_before_run?: boolean
  unlock_before_run?: boolean
  go_home_after_run?: boolean
}

export type ExecutionStatus = 'pending' | 'running' | 'success' | 'failed'

export interface Execution {
  id: number
  task_id: number
  task_name: string | null
  status: ExecutionStatus
  started_at: string | null
  finished_at: string | null
  error_message: string | null
}

export interface ExecutionStep {
  step: number
  action: string | Record<string, unknown> | null  // 支持字符串或对象
  thinking: string | null  // 思考内容
  description: string | null
  screenshot: string | null
  timestamp: string
}

export interface ExecutionDetail extends Execution {
  steps: ExecutionStep[] | null
  recording_path: string | null
  device_serial: string | null  // 执行任务的设备 serial
}

export type NotificationType = 'dingtalk' | 'telegram'

export interface NotificationChannel {
  id: number
  type: NotificationType
  name: string
  config: Record<string, string>
  enabled: boolean
}

export interface NotificationChannelCreate {
  type: NotificationType
  name: string
  config: Record<string, string>
  enabled?: boolean
}

export interface Device {
  serial: string
  status: string
  model: string | null
  product: string | null
}

export interface Settings {
  autoglm_base_url: string | null
  autoglm_api_key: string | null
  autoglm_model: string | null
  autoglm_max_steps: number | null
  selected_device: string | null  // 用户选择的设备 serial
}

export interface ConnectResponse {
  success: boolean
  message: string
  serial: string | null
}

// 系统提示词 - 针对设备配置
export interface SystemPrompt {
  id: number
  name: string
  description: string | null
  device_serial: string | null
  device_model: string | null
  system_prompt: string | null
  prefix_prompt: string | null
  suffix_prompt: string | null
  priority: number
  enabled: boolean
  created_at: string
  updated_at: string
}

export interface SystemPromptCreate {
  name: string
  description?: string
  device_serial?: string
  device_model?: string
  system_prompt?: string
  prefix_prompt?: string
  suffix_prompt?: string
  priority?: number
  enabled?: boolean
}

export interface SystemPromptUpdate {
  name?: string
  description?: string
  device_serial?: string
  device_model?: string
  system_prompt?: string
  prefix_prompt?: string
  suffix_prompt?: string
  priority?: number
  enabled?: boolean
}

// 任务模版
export interface TaskTemplate {
  id: number
  name: string
  description: string | null
  command: string
  default_cron: string | null
  category: string | null
  icon: string | null
  created_at: string
  updated_at: string
}

export interface TaskTemplateCreate {
  name: string
  description?: string
  command: string
  default_cron?: string
  category?: string
  icon?: string
}

export interface TaskTemplateUpdate {
  name?: string
  description?: string
  command?: string
  default_cron?: string
  category?: string
  icon?: string
}

// 自定义 APP 包名映射
export interface AppPackage {
  id: number
  app_name: string
  package_name: string
  created_at: string
  updated_at: string
}

export interface AppPackageCreate {
  app_name: string
  package_name: string
}

export interface AppPackageUpdate {
  app_name?: string
  package_name?: string
}

// 设备配置 - 唤醒和解锁设置
export interface DeviceConfig {
  id: number
  device_serial: string
  wake_enabled: boolean
  wake_command: string | null
  unlock_enabled: boolean
  unlock_type: 'swipe' | 'longpress' | null
  unlock_start_x: number | null
  unlock_start_y: number | null
  unlock_end_x: number | null
  unlock_end_y: number | null
  unlock_duration: number
  created_at: string
  updated_at: string
}

export interface DeviceConfigCreate {
  device_serial: string
  wake_enabled?: boolean
  wake_command?: string
  unlock_enabled?: boolean
  unlock_type?: 'swipe' | 'longpress'
  unlock_start_x?: number
  unlock_start_y?: number
  unlock_end_x?: number
  unlock_end_y?: number
  unlock_duration?: number
}

export interface DeviceConfigUpdate {
  wake_enabled?: boolean
  wake_command?: string
  unlock_enabled?: boolean
  unlock_type?: 'swipe' | 'longpress'
  unlock_start_x?: number
  unlock_start_y?: number
  unlock_end_x?: number
  unlock_end_y?: number
  unlock_duration?: number
}
