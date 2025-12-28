import { useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Switch } from '@/components/ui/switch'
import { Badge } from '@/components/ui/badge'
import { Checkbox } from '@/components/ui/checkbox'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
  DialogDescription,
} from '@/components/ui/dialog'
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
import { tasksApi, taskTemplatesApi, notificationsApi, executionsApi, devicesApi } from '@/api/client'
import { formatDateTime } from '@/lib/utils'
import type { Task, TaskCreate, TaskTemplate, Execution } from '@/types'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import {
  Plus,
  Play,
  Pencil,
  Trash2,
  Clock,
  FileCode,
  TreeDeciduous,
  Coffee,
  Gift,
  Zap,
  Bell,
  Eye,
  ChevronLeft,
  ChevronRight,
  ListTodo,
  History,
  Smartphone,
} from 'lucide-react'

// 图标映射
const iconMap: Record<string, React.ComponentType<{ className?: string }>> = {
  'tree-deciduous': TreeDeciduous,
  'coffee': Coffee,
  'gift': Gift,
  'zap': Zap,
}

// 执行历史组件
function ExecutionHistory() {
  const queryClient = useQueryClient()
  const navigate = useNavigate()
  const [page, setPage] = useState(0)
  const [showClearDialog, setShowClearDialog] = useState(false)
  const [deleteExecutionId, setDeleteExecutionId] = useState<number | null>(null)
  const pageSize = 10

  const { data: countData } = useQuery({
    queryKey: ['executions-count'],
    queryFn: () => executionsApi.count(),
  })

  const { data: executions = [], isLoading } = useQuery({
    queryKey: ['executions', page, pageSize],
    queryFn: () => executionsApi.list(undefined, pageSize, page * pageSize),
  })

  const deleteMutation = useMutation({
    mutationFn: executionsApi.delete,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['executions'] })
      queryClient.invalidateQueries({ queryKey: ['executions-count'] })
    },
  })

  const clearAllMutation = useMutation({
    mutationFn: executionsApi.clearAll,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['executions'] })
      queryClient.invalidateQueries({ queryKey: ['executions-count'] })
      setPage(0)
    },
  })

  const totalCount = countData?.count || 0
  const totalPages = Math.ceil(totalCount / pageSize)

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
    return <div className="py-8 text-center text-muted-foreground">加载中...</div>
  }

  return (
    <div className="space-y-4">
      {/* 工具栏 */}
      <div className="flex justify-between items-center">
        <div className="text-sm text-muted-foreground">
          共 {totalCount} 条记录
        </div>
        {totalCount > 0 && (
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowClearDialog(true)}
            disabled={clearAllMutation.isPending}
          >
            <Trash2 className="h-4 w-4 mr-2" />
            清空全部
          </Button>
        )}

        {/* 清空确认对话框 */}
        <AlertDialog open={showClearDialog} onOpenChange={setShowClearDialog}>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>确认清空</AlertDialogTitle>
              <AlertDialogDescription>
                确定要清空所有执行记录吗？此操作不可撤销。
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel>取消</AlertDialogCancel>
              <AlertDialogAction
                onClick={() => {
                  clearAllMutation.mutate()
                  setShowClearDialog(false)
                }}
                className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              >
                清空全部
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>

        {/* 删除单条记录确认对话框 */}
        <AlertDialog open={deleteExecutionId !== null} onOpenChange={(open) => !open && setDeleteExecutionId(null)}>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>确认删除</AlertDialogTitle>
              <AlertDialogDescription>
                确定要删除这条执行记录吗？此操作不可撤销。
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel>取消</AlertDialogCancel>
              <AlertDialogAction
                onClick={() => {
                  if (deleteExecutionId !== null) {
                    deleteMutation.mutate(deleteExecutionId)
                    setDeleteExecutionId(null)
                  }
                }}
                className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              >
                删除
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      </div>

      {/* 列表 */}
      {executions.length === 0 ? (
        <Card>
          <CardContent className="py-8 text-center">
            <p className="text-muted-foreground">暂无执行记录</p>
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-2">
          {executions.map((exec: Execution) => (
            <Card key={exec.id} className="hover:bg-muted/30 transition-colors">
              <CardContent className="py-3">
                <div className="flex items-center justify-between">
                  <div className="space-y-1 flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="font-medium truncate">
                        {exec.task_name || `任务 #${exec.task_id}`}
                      </span>
                      {getStatusBadge(exec.status)}
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {exec.started_at && (
                        <span>
                          {formatDateTime(exec.started_at)}
                        </span>
                      )}
                      {exec.finished_at && exec.started_at && (
                        <span className="ml-2">
                          耗时: {Math.round((new Date(exec.finished_at + 'Z').getTime() - new Date(exec.started_at + 'Z').getTime()) / 1000)}s
                        </span>
                      )}
                    </div>
                    {exec.error_message && (
                      <p className="text-xs text-destructive truncate">
                        {exec.error_message}
                      </p>
                    )}
                  </div>
                  <div className="flex gap-1 ml-2">
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => navigate(`/history/${exec.id}`)}
                    >
                      <Eye className="h-4 w-4" />
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => setDeleteExecutionId(exec.id)}
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* 分页 */}
      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setPage(p => Math.max(0, p - 1))}
            disabled={page === 0}
          >
            <ChevronLeft className="h-4 w-4" />
          </Button>
          <span className="text-sm text-muted-foreground">
            {page + 1} / {totalPages}
          </span>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))}
            disabled={page >= totalPages - 1}
          >
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>
      )}
    </div>
  )
}

// 任务列表组件
function TaskList() {
  const queryClient = useQueryClient()
  const navigate = useNavigate()
  const [isDialogOpen, setIsDialogOpen] = useState(false)
  const [isTemplateDialogOpen, setIsTemplateDialogOpen] = useState(false)
  const [editingTask, setEditingTask] = useState<Task | null>(null)
  const [deleteTaskId, setDeleteTaskId] = useState<number | null>(null)
  const [formData, setFormData] = useState<TaskCreate>({
    name: '',
    description: '',
    command: '',
    cron_expression: '0 8 * * *',
    enabled: true,
    notify_on_success: false,
    notify_on_failure: true,
    notification_channel_ids: null,
    auto_confirm_sensitive: true,
    random_delay_minutes: null,
    device_serial: null,
    wake_before_run: true,
    unlock_before_run: true,
  })

  const { data: tasks = [], isLoading } = useQuery({
    queryKey: ['tasks'],
    queryFn: tasksApi.list,
  })

  const { data: templates = [] } = useQuery({
    queryKey: ['task-templates'],
    queryFn: () => taskTemplatesApi.list(),
  })

  const { data: notificationChannels = [] } = useQuery({
    queryKey: ['notification-channels'],
    queryFn: notificationsApi.list,
  })

  const { data: devices = [] } = useQuery({
    queryKey: ['devices'],
    queryFn: devicesApi.list,
  })

  const createMutation = useMutation({
    mutationFn: tasksApi.create,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['tasks'] })
      closeDialog()
    },
  })

  const updateMutation = useMutation({
    mutationFn: ({ id, data }: { id: number; data: TaskCreate }) =>
      tasksApi.update(id, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['tasks'] })
      closeDialog()
    },
  })

  const deleteMutation = useMutation({
    mutationFn: tasksApi.delete,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['tasks'] })
    },
  })

  const runMutation = useMutation({
    mutationFn: tasksApi.run,
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['executions'] })
      navigate(`/history/${data.execution_id}`)
    },
  })

  const openCreateDialog = () => {
    setEditingTask(null)
    setFormData({
      name: '',
      description: '',
      command: '',
      cron_expression: '0 8 * * *',
      enabled: true,
      notify_on_success: false,
      notify_on_failure: true,
      notification_channel_ids: null,
      auto_confirm_sensitive: true,
      device_serial: null,
      random_delay_minutes: null,
      wake_before_run: true,
      unlock_before_run: true,
      go_home_after_run: false,
      max_steps: null,
    })
    setIsDialogOpen(true)
  }

  const openTemplateDialog = () => {
    setIsTemplateDialogOpen(true)
  }

  const selectTemplate = (template: TaskTemplate) => {
    setFormData({
      name: template.name,
      description: template.description || '',
      command: template.command,
      cron_expression: '0 8 * * *',
      enabled: true,
      notify_on_success: false,
      notify_on_failure: true,
      notification_channel_ids: null,
      auto_confirm_sensitive: true,
      device_serial: null,
      random_delay_minutes: null,
      wake_before_run: true,
      unlock_before_run: true,
      go_home_after_run: false,
      max_steps: null,
    })
    setIsTemplateDialogOpen(false)
    setIsDialogOpen(true)
  }

  const openEditDialog = (task: Task) => {
    setEditingTask(task)
    setFormData({
      name: task.name,
      description: task.description || '',
      command: task.command,
      cron_expression: task.cron_expression,
      enabled: task.enabled,
      notify_on_success: task.notify_on_success,
      notify_on_failure: task.notify_on_failure,
      notification_channel_ids: task.notification_channel_ids,
      auto_confirm_sensitive: task.auto_confirm_sensitive,
      device_serial: task.device_serial,
      random_delay_minutes: task.random_delay_minutes,
      wake_before_run: task.wake_before_run,
      unlock_before_run: task.unlock_before_run,
      go_home_after_run: task.go_home_after_run,
      max_steps: task.max_steps,
    })
    setIsDialogOpen(true)
  }

  const closeDialog = () => {
    setIsDialogOpen(false)
    setEditingTask(null)
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    // 自动添加浏览器时区
    const dataWithTimezone = {
      ...formData,
      timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
    }
    if (editingTask) {
      updateMutation.mutate({ id: editingTask.id, data: dataWithTimezone })
    } else {
      createMutation.mutate(dataWithTimezone)
    }
  }

  const cronPresets = [
    { label: '每天8点', value: '0 8 * * *' },
    { label: '每天7点', value: '0 7 * * *' },
    { label: '每小时', value: '0 * * * *' },
    { label: '每30分钟', value: '*/30 * * * *' },
    { label: '工作日9点', value: '0 9 * * 1-5' },
  ]

  if (isLoading) {
    return <div className="py-8 text-center text-muted-foreground">加载中...</div>
  }

  return (
    <div className="space-y-4">
      <div className="flex justify-end gap-2">
        {templates.length > 0 && (
          <Button variant="outline" onClick={openTemplateDialog}>
            <FileCode className="h-4 w-4 mr-2" />
            从模版创建
          </Button>
        )}
        <Button onClick={openCreateDialog}>
          <Plus className="h-4 w-4 mr-2" />
          创建任务
        </Button>
      </div>

      {tasks.length === 0 ? (
        <Card>
          <CardContent className="py-8 text-center">
            <p className="text-muted-foreground mb-4">暂无任务</p>
            <div className="flex gap-2 justify-center">
              {templates.length > 0 && (
                <Button variant="outline" onClick={openTemplateDialog}>
                  从模版创建
                </Button>
              )}
              <Button onClick={openCreateDialog}>创建第一个任务</Button>
            </div>
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-4">
          {tasks.map((task) => (
            <Card key={task.id}>
              <CardHeader className="pb-3">
                <div className="flex items-start justify-between">
                  <div className="space-y-1">
                    <CardTitle className="flex items-center gap-2">
                      {task.name}
                      <Badge variant={task.enabled ? 'success' : 'secondary'}>
                        {task.enabled ? '启用' : '禁用'}
                      </Badge>
                    </CardTitle>
                    {task.description && (
                      <p className="text-sm text-muted-foreground">
                        {task.description}
                      </p>
                    )}
                  </div>
                  <div className="flex gap-2">
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => runMutation.mutate(task.id)}
                      disabled={runMutation.isPending}
                    >
                      <Play className="h-4 w-4" />
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => openEditDialog(task)}
                    >
                      <Pencil className="h-4 w-4" />
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => setDeleteTaskId(task.id)}
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="grid gap-2 text-sm">
                  <div className="flex items-center gap-2">
                    <span className="text-muted-foreground">指令:</span>
                    <code className="bg-muted px-2 py-1 rounded text-xs">
                      {task.command}
                    </code>
                  </div>
                  <div className="flex items-center gap-2 flex-wrap">
                    <Clock className="h-4 w-4 text-muted-foreground" />
                    <span className="text-muted-foreground">Cron:</span>
                    <code className="bg-muted px-2 py-1 rounded text-xs">
                      {task.cron_expression}
                    </code>
                    {task.random_delay_minutes && task.random_delay_minutes > 0 && (
                      <span className="text-xs text-orange-500">
                        (随机延迟 0~{task.random_delay_minutes} 分钟)
                      </span>
                    )}
                    {task.next_run && (
                      <span className="text-muted-foreground">
                        下次运行: {formatDateTime(task.next_run)}
                      </span>
                    )}
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Template Selection Dialog */}
      <Dialog open={isTemplateDialogOpen} onOpenChange={setIsTemplateDialogOpen}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>选择任务模版</DialogTitle>
            <DialogDescription>
              选择一个模版快速创建任务
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 md:grid-cols-2 max-h-[60vh] overflow-y-auto">
            {templates.map((template) => {
              const IconComponent = template.icon ? iconMap[template.icon] : FileCode
              return (
                <Card
                  key={template.id}
                  className="cursor-pointer hover:bg-accent/50 transition-colors"
                  onClick={() => selectTemplate(template)}
                >
                  <CardHeader className="pb-2">
                    <CardTitle className="flex items-center gap-2 text-base">
                      {IconComponent && <IconComponent className="h-5 w-5" />}
                      {template.name}
                    </CardTitle>
                    {template.description && (
                      <CardDescription className="text-xs">
                        {template.description}
                      </CardDescription>
                    )}
                  </CardHeader>
                  <CardContent>
                    <div className="text-xs text-muted-foreground space-y-1">
                      <p className="line-clamp-2">{template.command}</p>
                      {template.default_cron && (
                        <p>
                          <span className="text-muted-foreground">定时: </span>
                          <code className="bg-muted px-1 rounded">
                            {template.default_cron}
                          </code>
                        </p>
                      )}
                    </div>
                  </CardContent>
                </Card>
              )
            })}
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setIsTemplateDialogOpen(false)}>
              取消
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Create/Edit Dialog */}
      <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>
              {editingTask ? '编辑任务' : '创建任务'}
            </DialogTitle>
          </DialogHeader>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="name">任务名称</Label>
              <Input
                id="name"
                value={formData.name}
                onChange={(e) =>
                  setFormData({ ...formData, name: e.target.value })
                }
                placeholder="如: 蚂蚁森林收能量"
                required
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="description">描述 (可选)</Label>
              <Input
                id="description"
                value={formData.description}
                onChange={(e) =>
                  setFormData({ ...formData, description: e.target.value })
                }
                placeholder="任务描述"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="command">AutoGLM 指令</Label>
              <Textarea
                id="command"
                value={formData.command}
                onChange={(e) =>
                  setFormData({ ...formData, command: e.target.value })
                }
                placeholder="如: 打开支付宝，进入蚂蚁森林，收取所有好友的能量"
                required
                rows={3}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="cron">Cron 表达式</Label>
              <div className="flex gap-2">
                <Input
                  id="cron"
                  value={formData.cron_expression}
                  onChange={(e) =>
                    setFormData({ ...formData, cron_expression: e.target.value })
                  }
                  placeholder="0 8 * * *"
                  required
                />
              </div>
              <div className="flex flex-wrap gap-2 mt-2">
                {cronPresets.map((preset) => (
                  <Button
                    key={preset.value}
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={() =>
                      setFormData({ ...formData, cron_expression: preset.value })
                    }
                  >
                    {preset.label}
                  </Button>
                ))}
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Label htmlFor="random_delay">随机延迟 (分钟)</Label>
                <span className="text-xs text-muted-foreground">
                  (在 Cron 触发后随机延迟执行，增加随机性)
                </span>
              </div>
              <div className="flex gap-2 items-center">
                <Input
                  id="random_delay"
                  type="number"
                  min="0"
                  max="120"
                  value={formData.random_delay_minutes ?? ''}
                  onChange={(e) =>
                    setFormData({
                      ...formData,
                      random_delay_minutes: e.target.value ? parseInt(e.target.value, 10) : null,
                    })
                  }
                  placeholder="不延迟"
                  className="w-24"
                />
                <span className="text-sm text-muted-foreground">
                  {formData.random_delay_minutes
                    ? `将在 0~${formData.random_delay_minutes} 分钟内随机执行`
                    : '准时执行'}
                </span>
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Label htmlFor="max_steps">最大步数</Label>
                <span className="text-xs text-muted-foreground">
                  (留空则使用全局设置)
                </span>
              </div>
              <div className="flex gap-2 items-center">
                <Input
                  id="max_steps"
                  type="number"
                  min="1"
                  max="100"
                  value={formData.max_steps ?? ''}
                  onChange={(e) =>
                    setFormData({
                      ...formData,
                      max_steps: e.target.value ? parseInt(e.target.value, 10) : null,
                    })
                  }
                  placeholder="使用全局设置"
                  className="w-32"
                />
                <span className="text-sm text-muted-foreground">
                  {formData.max_steps
                    ? `最多执行 ${formData.max_steps} 步`
                    : '使用全局设置'}
                </span>
              </div>
            </div>

            <div className="flex items-center justify-between">
              <Label htmlFor="enabled">启用任务</Label>
              <Switch
                id="enabled"
                checked={formData.enabled}
                onCheckedChange={(checked) =>
                  setFormData({ ...formData, enabled: checked })
                }
              />
            </div>

            <div className="flex items-center justify-between">
              <div>
                <Label htmlFor="auto_confirm_sensitive">敏感操作自动确认</Label>
                <p className="text-xs text-muted-foreground mt-0.5">
                  关闭后遇到重要操作会暂停等待手动确认
                </p>
              </div>
              <Switch
                id="auto_confirm_sensitive"
                checked={formData.auto_confirm_sensitive}
                onCheckedChange={(checked) =>
                  setFormData({ ...formData, auto_confirm_sensitive: checked })
                }
              />
            </div>

            {/* 执行设备选择 */}
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Smartphone className="h-4 w-4 text-muted-foreground" />
                <Label>执行设备</Label>
                <span className="text-xs text-muted-foreground">
                  (不选择则使用全局设置的设备)
                </span>
              </div>
              <Select
                value={formData.device_serial || '__global__'}
                onValueChange={(value) =>
                  setFormData({
                    ...formData,
                    device_serial: value === '__global__' ? null : value,
                  })
                }
              >
                <SelectTrigger>
                  <SelectValue placeholder="使用全局设置的设备" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="__global__">使用全局设置的设备</SelectItem>
                  {devices
                    .filter((d) => d.status === 'device')
                    .map((device) => (
                      <SelectItem key={device.serial} value={device.serial}>
                        {device.model || device.serial}
                        <span className="text-xs text-muted-foreground ml-2">
                          ({device.serial})
                        </span>
                      </SelectItem>
                    ))}
                </SelectContent>
              </Select>

              {/* 唤醒和解锁选项 */}
              <div className="pl-6 space-y-3 pt-2">
                <div className="flex items-center justify-between">
                  <div>
                    <Label htmlFor="wake_before_run">执行前唤醒设备</Label>
                    <p className="text-xs text-muted-foreground mt-0.5">
                      使用设备配置中的唤醒命令
                    </p>
                  </div>
                  <Switch
                    id="wake_before_run"
                    checked={formData.wake_before_run}
                    onCheckedChange={(checked) =>
                      setFormData({ ...formData, wake_before_run: checked })
                    }
                  />
                </div>
                <div className="flex items-center justify-between">
                  <div>
                    <Label htmlFor="unlock_before_run">执行前解锁设备</Label>
                    <p className="text-xs text-muted-foreground mt-0.5">
                      使用设备配置中的解锁命令
                    </p>
                  </div>
                  <Switch
                    id="unlock_before_run"
                    checked={formData.unlock_before_run}
                    onCheckedChange={(checked) =>
                      setFormData({ ...formData, unlock_before_run: checked })
                    }
                  />
                </div>
                <div className="flex items-center justify-between">
                  <div>
                    <Label htmlFor="go_home_after_run">执行后返回主屏幕</Label>
                    <p className="text-xs text-muted-foreground mt-0.5">
                      任务完成后自动返回桌面
                    </p>
                  </div>
                  <Switch
                    id="go_home_after_run"
                    checked={formData.go_home_after_run}
                    onCheckedChange={(checked) =>
                      setFormData({ ...formData, go_home_after_run: checked })
                    }
                  />
                </div>
              </div>
            </div>

            <div className="flex items-center justify-between">
              <Label htmlFor="notify_success">成功时通知</Label>
              <Switch
                id="notify_success"
                checked={formData.notify_on_success}
                onCheckedChange={(checked) =>
                  setFormData({ ...formData, notify_on_success: checked })
                }
              />
            </div>

            <div className="flex items-center justify-between">
              <Label htmlFor="notify_failure">失败时通知</Label>
              <Switch
                id="notify_failure"
                checked={formData.notify_on_failure}
                onCheckedChange={(checked) =>
                  setFormData({ ...formData, notify_on_failure: checked })
                }
              />
            </div>

            {/* 通知渠道选择 */}
            {(formData.notify_on_success || formData.notify_on_failure) &&
              notificationChannels.length > 0 && (
                <div className="space-y-3">
                  <div className="flex items-center gap-2">
                    <Bell className="h-4 w-4 text-muted-foreground" />
                    <Label>通知渠道</Label>
                    <span className="text-xs text-muted-foreground">
                      (不选择则使用所有启用的渠道)
                    </span>
                  </div>
                  <div className="space-y-2 pl-6">
                    {notificationChannels
                      .filter((c) => c.enabled)
                      .map((channel) => (
                        <div key={channel.id} className="flex items-center gap-2">
                          <Checkbox
                            id={`channel-${channel.id}`}
                            checked={
                              formData.notification_channel_ids?.includes(channel.id) ??
                              false
                            }
                            onCheckedChange={(checked) => {
                              const currentIds = formData.notification_channel_ids || []
                              if (checked) {
                                setFormData({
                                  ...formData,
                                  notification_channel_ids: [...currentIds, channel.id],
                                })
                              } else {
                                const newIds = currentIds.filter((id) => id !== channel.id)
                                setFormData({
                                  ...formData,
                                  notification_channel_ids:
                                    newIds.length > 0 ? newIds : null,
                                })
                              }
                            }}
                          />
                          <label
                            htmlFor={`channel-${channel.id}`}
                            className="text-sm cursor-pointer"
                          >
                            <span className="font-medium">{channel.name}</span>
                            <Badge variant="outline" className="ml-2 text-xs">
                              {channel.type}
                            </Badge>
                          </label>
                        </div>
                      ))}
                  </div>
                </div>
              )}

            <DialogFooter>
              <Button type="button" variant="outline" onClick={closeDialog}>
                取消
              </Button>
              <Button
                type="submit"
                disabled={createMutation.isPending || updateMutation.isPending}
              >
                {editingTask ? '保存' : '创建'}
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>

      {/* 删除任务确认对话框 */}
      <AlertDialog open={deleteTaskId !== null} onOpenChange={(open) => !open && setDeleteTaskId(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>确认删除</AlertDialogTitle>
            <AlertDialogDescription>
              确定要删除这个任务吗？相关的执行记录也会被删除，此操作不可撤销。
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>取消</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => {
                if (deleteTaskId !== null) {
                  deleteMutation.mutate(deleteTaskId)
                  setDeleteTaskId(null)
                }
              }}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              删除
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  )
}

export function Tasks() {
  const [searchParams, setSearchParams] = useSearchParams()
  const currentTab = searchParams.get('tab') || 'tasks'

  const handleTabChange = (value: string) => {
    setSearchParams({ tab: value })
  }

  return (
    <div className="space-y-6 h-full flex flex-col">
      <h1 className="text-3xl font-bold">任务管理</h1>

      <Tabs value={currentTab} onValueChange={handleTabChange} className="flex-1 flex flex-col min-h-0">
        <TabsList>
          <TabsTrigger value="tasks">
            <ListTodo className="h-4 w-4 mr-2" />
            任务列表
          </TabsTrigger>
          <TabsTrigger value="history">
            <History className="h-4 w-4 mr-2" />
            执行历史
          </TabsTrigger>
        </TabsList>
        <TabsContent value="tasks" className="flex-1 overflow-auto">
          <TaskList />
        </TabsContent>
        <TabsContent value="history" className="flex-1 overflow-auto">
          <ExecutionHistory />
        </TabsContent>
      </Tabs>
    </div>
  )
}
