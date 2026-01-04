import axios from 'axios'
import type {
  Task,
  TaskCreate,
  TaskUpdate,
  Execution,
  ExecutionDetail,
  NotificationChannel,
  NotificationChannelCreate,
  Device,
  Settings,
  ConnectResponse,
  SystemPrompt,
  SystemPromptCreate,
  SystemPromptUpdate,
  TaskTemplate,
  TaskTemplateCreate,
  TaskTemplateUpdate,
  AppPackage,
  AppPackageCreate,
  AppPackageUpdate,
  DeviceConfig,
  DeviceConfigUpdate,
  PairRequest,
  RegisteredDevice,
  QRCodePairingResponse,
  PairingStatusResponse,
} from '@/types'

const api = axios.create({
  baseURL: '/api',
})

// Tasks
export const tasksApi = {
  list: () => api.get<Task[]>('/tasks').then(r => r.data),
  get: (id: number) => api.get<Task>(`/tasks/${id}`).then(r => r.data),
  create: (data: TaskCreate) => api.post<Task>('/tasks', data).then(r => r.data),
  update: (id: number, data: TaskUpdate) => api.put<Task>(`/tasks/${id}`, data).then(r => r.data),
  delete: (id: number) => api.delete(`/tasks/${id}`),
  run: (id: number) => api.post<{ message: string; execution_id: number }>(`/tasks/${id}/run`).then(r => r.data),
}

// Executions
export const executionsApi = {
  list: (taskId?: number, limit = 50, offset = 0) =>
    api.get<Execution[]>('/executions', {
      params: { task_id: taskId, limit, offset },
    }).then(r => r.data),
  get: (id: number) => api.get<ExecutionDetail>(`/executions/${id}`).then(r => r.data),
  delete: (id: number) => api.delete(`/executions/${id}`),
  clearAll: () => api.delete('/executions').then(r => r.data),
  count: (taskId?: number) =>
    api.get<{ count: number }>('/executions/count', {
      params: taskId ? { task_id: taskId } : undefined,
    }).then(r => r.data),
  getRecordingUrl: (id: number) => `/api/executions/${id}/recording`,
  getStreamUrl: (id: number) => `/api/executions/${id}/stream`,
}

// Notifications
export const notificationsApi = {
  list: () => api.get<NotificationChannel[]>('/notifications').then(r => r.data),
  get: (id: number) => api.get<NotificationChannel>(`/notifications/${id}`).then(r => r.data),
  create: (data: NotificationChannelCreate) =>
    api.post<NotificationChannel>('/notifications', data).then(r => r.data),
  update: (id: number, data: Partial<NotificationChannelCreate>) =>
    api.put<NotificationChannel>(`/notifications/${id}`, data).then(r => r.data),
  delete: (id: number) => api.delete(`/notifications/${id}`),
  test: (id: number) => api.post(`/notifications/${id}/test`),
}

// 设备忙碌状态
export interface DeviceBusyStatus {
  is_busy: boolean
  execution_id: number | null
  task_id: number | null
  task_name: string | null
  started_at: string | null
}

// Devices
export const devicesApi = {
  list: () => api.get<Device[]>('/devices').then(r => r.data),
  refresh: () => api.post<Device[]>('/devices/refresh').then(r => r.data),
  connect: (address: string, register = true) =>
    api.post<ConnectResponse>('/devices/connect', { address, register }).then(r => r.data),
  disconnect: (serial: string) =>
    api.post<ConnectResponse>(`/devices/disconnect/${encodeURIComponent(serial)}`).then(r => r.data),
  getStreamUrl: (serial: string) => `${api.defaults.baseURL}/devices/${serial}/stream`,
  getScreenshotUrl: (serial: string) => `${api.defaults.baseURL}/devices/${serial}/screenshot`,
  // 设备控制
  sendKeyEvent: (serial: string, key: 'home' | 'back' | 'app_switch') =>
    api.post(`/devices/${encodeURIComponent(serial)}/keyevent`, { key }).then(r => r.data),
  sendSwipe: (serial: string, startX: number, startY: number, endX: number, endY: number, duration = 300) =>
    api.post(`/devices/${encodeURIComponent(serial)}/swipe`, {
      start_x: startX, start_y: startY, end_x: endX, end_y: endY, duration,
    }).then(r => r.data),
  sendTap: (serial: string, x: number, y: number) =>
    api.post(`/devices/${encodeURIComponent(serial)}/tap`, { x, y }).then(r => r.data),
  // 设备忙碌状态
  getBusyStatus: (serial: string) =>
    api.get<DeviceBusyStatus>(`/devices/${encodeURIComponent(serial)}/busy-status`).then(r => r.data),
  release: (serial: string) =>
    api.post<{ success: boolean; message: string; released_count: number }>(
      `/devices/${encodeURIComponent(serial)}/release`
    ).then(r => r.data),
  // WiFi 设备配对和重连
  pair: (data: PairRequest) =>
    api.post<ConnectResponse>('/devices/pair', data).then(r => r.data),
  getPairingQRCode: () =>
    api.get<QRCodePairingResponse>('/devices/pair/qrcode').then(r => r.data),
  getPairingStatus: (sessionId: string) =>
    api.get<PairingStatusResponse>(`/devices/pair/status/${sessionId}`).then(r => r.data),
  cancelPairing: (sessionId: string) =>
    api.delete<{ success: boolean; message: string }>(`/devices/pair/${sessionId}`).then(r => r.data),
  reconnect: (serial: string) =>
    api.post<{ success: boolean; message: string }>(`/devices/reconnect/${encodeURIComponent(serial)}`).then(r => r.data),
  // 已注册设备管理
  getRegistered: () => api.get<RegisteredDevice[]>('/devices/registered').then(r => r.data),
  unregister: (serial: string) =>
    api.delete<{ success: boolean; message: string }>(`/devices/registered/${encodeURIComponent(serial)}`).then(r => r.data),
}

// Settings
export const settingsApi = {
  get: () => api.get<Settings>('/settings').then(r => r.data),
  update: (data: Partial<Settings>) => api.put<Settings>('/settings', data).then(r => r.data),
  test: () => api.post<{ success: boolean; message: string; models: string[] | null }>('/settings/test').then(r => r.data),
}

// Health
export const healthApi = {
  check: () => api.get('/health').then(r => r.data),
}

// System Prompts (设备系统提示词)
export const systemPromptsApi = {
  list: () => api.get<SystemPrompt[]>('/system-prompts').then(r => r.data),
  get: (id: number) => api.get<SystemPrompt>(`/system-prompts/${id}`).then(r => r.data),
  create: (data: SystemPromptCreate) =>
    api.post<SystemPrompt>('/system-prompts', data).then(r => r.data),
  update: (id: number, data: SystemPromptUpdate) =>
    api.put<SystemPrompt>(`/system-prompts/${id}`, data).then(r => r.data),
  delete: (id: number) => api.delete(`/system-prompts/${id}`),
  match: (deviceSerial: string, deviceModel?: string) =>
    api.get<SystemPrompt[]>(`/system-prompts/match/${deviceSerial}`, {
      params: { device_model: deviceModel },
    }).then(r => r.data),
  preview: (deviceSerial: string, command: string, deviceModel?: string) =>
    api.post<{ original_command: string; system_prompt: string; final_command: string }>(
      '/system-prompts/preview',
      null,
      { params: { device_serial: deviceSerial, command, device_model: deviceModel } }
    ).then(r => r.data),
}

// Task Templates (任务模版)
export const taskTemplatesApi = {
  list: (category?: string) =>
    api.get<TaskTemplate[]>('/task-templates', {
      params: category ? { category } : undefined,
    }).then(r => r.data),
  get: (id: number) => api.get<TaskTemplate>(`/task-templates/${id}`).then(r => r.data),
  create: (data: TaskTemplateCreate) =>
    api.post<TaskTemplate>('/task-templates', data).then(r => r.data),
  update: (id: number, data: TaskTemplateUpdate) =>
    api.put<TaskTemplate>(`/task-templates/${id}`, data).then(r => r.data),
  delete: (id: number) => api.delete(`/task-templates/${id}`),
  categories: () => api.get<string[]>('/task-templates/categories').then(r => r.data),
}

// App Packages (自定义 APP 包名)
export const appPackagesApi = {
  list: () => api.get<AppPackage[]>('/app-packages').then(r => r.data),
  get: (id: number) => api.get<AppPackage>(`/app-packages/${id}`).then(r => r.data),
  create: (data: AppPackageCreate) =>
    api.post<AppPackage>('/app-packages', data).then(r => r.data),
  update: (id: number, data: AppPackageUpdate) =>
    api.put<AppPackage>(`/app-packages/${id}`, data).then(r => r.data),
  delete: (id: number) => api.delete(`/app-packages/${id}`),
}

// Device Configs (设备唤醒解锁配置)
export const deviceConfigsApi = {
  list: () => api.get<DeviceConfig[]>('/device-configs').then(r => r.data),
  get: (serial: string) => api.get<DeviceConfig | null>(`/device-configs/${encodeURIComponent(serial)}`).then(r => r.data),
  update: (serial: string, data: DeviceConfigUpdate) =>
    api.put<DeviceConfig>(`/device-configs/${encodeURIComponent(serial)}`, data).then(r => r.data),
  delete: (serial: string) => api.delete(`/device-configs/${encodeURIComponent(serial)}`),
  getScreenStatus: (serial: string) =>
    api.get<{ screen_on: boolean; is_locked: boolean }>(`/device-configs/${encodeURIComponent(serial)}/screen-status`).then(r => r.data),
  testWake: (serial: string, data: { wake_command?: string }) =>
    api.post<{ success: boolean; message: string }>(`/device-configs/${encodeURIComponent(serial)}/test-wake`, data).then(r => r.data),
  testUnlock: (serial: string, data: {
    unlock_type: string
    unlock_start_x: number
    unlock_start_y: number
    unlock_end_x?: number
    unlock_end_y?: number
    unlock_duration: number
  }) =>
    api.post<{ success: boolean; message: string }>(`/device-configs/${encodeURIComponent(serial)}/test-unlock`, data).then(r => r.data),
  lockScreen: (serial: string) =>
    api.post<{ success: boolean; message: string }>(`/device-configs/${encodeURIComponent(serial)}/lock`).then(r => r.data),
}

// Debug (调试控制台)
export const debugApi = {
  stop: (serial: string) =>
    api.post<{ success: boolean; message: string }>(`/debug/stop/${encodeURIComponent(serial)}`).then(r => r.data),
  getStatus: (serial: string) =>
    api.get<{ is_running: boolean }>(`/debug/status/${encodeURIComponent(serial)}`).then(r => r.data),
}
