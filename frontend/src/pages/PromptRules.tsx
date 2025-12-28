import { useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Switch } from '@/components/ui/switch'
import { Badge } from '@/components/ui/badge'
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
import { systemPromptsApi, taskTemplatesApi, devicesApi } from '@/api/client'
import type { SystemPrompt, SystemPromptCreate, TaskTemplate, TaskTemplateCreate } from '@/types'
import {
  Plus,
  Pencil,
  Trash2,
  Settings2,
  FileCode,
  TreeDeciduous,
  Coffee,
  Gift,
  Zap,
} from 'lucide-react'

// 图标映射
const iconMap: Record<string, React.ComponentType<{ className?: string }>> = {
  'tree-deciduous': TreeDeciduous,
  'coffee': Coffee,
  'gift': Gift,
  'zap': Zap,
}

export function PromptRules() {
  const queryClient = useQueryClient()
  const [searchParams, setSearchParams] = useSearchParams()
  const activeTab = searchParams.get('tab') || 'task-templates'

  const handleTabChange = (value: string) => {
    setSearchParams({ tab: value })
  }

  // ======== 系统提示词状态 ========
  const [isPromptDialogOpen, setIsPromptDialogOpen] = useState(false)
  const [editingPrompt, setEditingPrompt] = useState<SystemPrompt | null>(null)
  const [promptFormData, setPromptFormData] = useState<SystemPromptCreate>({
    name: '',
    description: '',
    device_serial: '',
    device_model: '',
    system_prompt: '',
    prefix_prompt: '',
    suffix_prompt: '',
    priority: 0,
    enabled: true,
  })

  const [deletePromptId, setDeletePromptId] = useState<number | null>(null)

  // ======== 任务模版状态 ========
  const [isTemplateDialogOpen, setIsTemplateDialogOpen] = useState(false)
  const [editingTemplate, setEditingTemplate] = useState<TaskTemplate | null>(null)
  const [deleteTemplateId, setDeleteTemplateId] = useState<number | null>(null)
  const [templateFormData, setTemplateFormData] = useState<TaskTemplateCreate>({
    name: '',
    description: '',
    command: '',
    default_cron: '',
    category: '',
    icon: '',
  })

  // ======== 查询数据 ========
  const { data: systemPrompts = [], isLoading: promptsLoading } = useQuery({
    queryKey: ['system-prompts'],
    queryFn: systemPromptsApi.list,
  })

  const { data: taskTemplates = [], isLoading: templatesLoading } = useQuery({
    queryKey: ['task-templates'],
    queryFn: () => taskTemplatesApi.list(),
  })

  useQuery({
    queryKey: ['devices'],
    queryFn: devicesApi.list,
  })

  // ======== 系统提示词 Mutations ========
  const createPromptMutation = useMutation({
    mutationFn: systemPromptsApi.create,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['system-prompts'] })
      closePromptDialog()
    },
  })

  const updatePromptMutation = useMutation({
    mutationFn: ({ id, data }: { id: number; data: SystemPromptCreate }) =>
      systemPromptsApi.update(id, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['system-prompts'] })
      closePromptDialog()
    },
  })

  const deletePromptMutation = useMutation({
    mutationFn: systemPromptsApi.delete,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['system-prompts'] })
    },
  })

  // ======== 任务模版 Mutations ========
  const createTemplateMutation = useMutation({
    mutationFn: taskTemplatesApi.create,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['task-templates'] })
      closeTemplateDialog()
    },
  })

  const updateTemplateMutation = useMutation({
    mutationFn: ({ id, data }: { id: number; data: TaskTemplateCreate }) =>
      taskTemplatesApi.update(id, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['task-templates'] })
      closeTemplateDialog()
    },
  })

  const deleteTemplateMutation = useMutation({
    mutationFn: taskTemplatesApi.delete,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['task-templates'] })
    },
  })

  // ======== 系统提示词 Dialog 函数 ========
  const openCreatePromptDialog = () => {
    setEditingPrompt(null)
    setPromptFormData({
      name: '',
      description: '',
      device_serial: '',
      device_model: '',
      system_prompt: '',
      prefix_prompt: '',
      suffix_prompt: '',
      priority: 0,
      enabled: true,
    })
    setIsPromptDialogOpen(true)
  }

  const openEditPromptDialog = (prompt: SystemPrompt) => {
    setEditingPrompt(prompt)
    setPromptFormData({
      name: prompt.name,
      description: prompt.description || '',
      device_serial: prompt.device_serial || '',
      device_model: prompt.device_model || '',
      system_prompt: prompt.system_prompt || '',
      prefix_prompt: prompt.prefix_prompt || '',
      suffix_prompt: prompt.suffix_prompt || '',
      priority: prompt.priority,
      enabled: prompt.enabled,
    })
    setIsPromptDialogOpen(true)
  }

  const closePromptDialog = () => {
    setIsPromptDialogOpen(false)
    setEditingPrompt(null)
  }

  const handlePromptSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (editingPrompt) {
      updatePromptMutation.mutate({ id: editingPrompt.id, data: promptFormData })
    } else {
      createPromptMutation.mutate(promptFormData)
    }
  }

  // ======== 任务模版 Dialog 函数 ========
  const openCreateTemplateDialog = () => {
    setEditingTemplate(null)
    setTemplateFormData({
      name: '',
      description: '',
      command: '',
      default_cron: '',
      category: '',
      icon: '',
    })
    setIsTemplateDialogOpen(true)
  }

  const openEditTemplateDialog = (template: TaskTemplate) => {
    setEditingTemplate(template)
    setTemplateFormData({
      name: template.name,
      description: template.description || '',
      command: template.command,
      default_cron: template.default_cron || '',
      category: template.category || '',
      icon: template.icon || '',
    })
    setIsTemplateDialogOpen(true)
  }

  const closeTemplateDialog = () => {
    setIsTemplateDialogOpen(false)
    setEditingTemplate(null)
  }

  const handleTemplateSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (editingTemplate) {
      updateTemplateMutation.mutate({ id: editingTemplate.id, data: templateFormData })
    } else {
      createTemplateMutation.mutate(templateFormData)
    }
  }

  // ======== 预设模版 ========
  const presetTemplates = [
    {
      name: '蚂蚁森林收能量',
      description: '每天早上自动收取蚂蚁森林能量',
      command: '打开支付宝，进入蚂蚁森林，收取所有可收取的能量，然后返回首页',
      category: '日常任务',
      icon: 'tree-deciduous',
    },
    {
      name: '京东签到',
      description: '京东 APP 每日签到领京豆',
      command: '打开京东APP，进入签到页面，完成每日签到，领取京豆奖励',
      category: '签到',
      icon: 'gift',
    },
    {
      name: '美团外卖点餐',
      description: '定时打开美团点早餐',
      command: '打开美团外卖，搜索早餐，选择第一个店铺，加购一份豆浆油条套餐',
      category: '点餐',
      icon: 'coffee',
    },
  ]

  const applyPresetTemplate = (preset: typeof presetTemplates[0]) => {
    setTemplateFormData({
      name: preset.name,
      description: preset.description,
      command: preset.command,
      default_cron: '',
      category: preset.category,
      icon: preset.icon,
    })
  }

  // ======== 设备预设 ========
  const devicePresets = [
    {
      name: '华为手机通用',
      device_model: 'HUAWEI*',
      prefix_prompt: '这是华为手机，使用HarmonyOS系统。',
    },
    {
      name: '小米手机通用',
      device_model: 'Mi*',
      prefix_prompt: '这是小米手机，使用MIUI系统。',
    },
    {
      name: 'OPPO手机通用',
      device_model: 'OPPO*',
      prefix_prompt: '这是OPPO手机，使用ColorOS系统。',
    },
  ]

  const applyDevicePreset = (preset: typeof devicePresets[0]) => {
    setPromptFormData({
      ...promptFormData,
      name: preset.name,
      device_model: preset.device_model || '',
      prefix_prompt: preset.prefix_prompt || '',
    })
  }

  if (promptsLoading || templatesLoading) {
    return <div>加载中...</div>
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold">Prompt 配置</h1>
        <p className="text-muted-foreground mt-1">
          配置系统提示词和任务模版
        </p>
      </div>

      <Tabs value={activeTab} onValueChange={handleTabChange}>
        <TabsList className="grid w-full grid-cols-2 max-w-md">
          <TabsTrigger value="task-templates" className="flex items-center gap-2">
            <FileCode className="h-4 w-4" />
            任务模版
          </TabsTrigger>
          <TabsTrigger value="system-prompts" className="flex items-center gap-2">
            <Settings2 className="h-4 w-4" />
            系统提示词
          </TabsTrigger>
        </TabsList>

        {/* ========== 系统提示词 Tab ========== */}
        <TabsContent value="system-prompts" className="space-y-6">
          <div className="flex justify-between items-center">
            <p className="text-sm text-muted-foreground">
              针对不同设备配置系统级提示词，自动应用于任务执行
            </p>
            <Button onClick={openCreatePromptDialog}>
              <Plus className="h-4 w-4 mr-2" />
              创建提示词
            </Button>
          </div>

          {/* 提示词列表 */}
          {systemPrompts.length === 0 ? (
            <Card>
              <CardContent className="py-8 text-center">
                <Settings2 className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                <p className="text-muted-foreground mb-4">暂无系统提示词</p>
                <Button onClick={openCreatePromptDialog}>创建第一个提示词</Button>
              </CardContent>
            </Card>
          ) : (
            <div className="grid gap-4">
              {systemPrompts.map((prompt) => (
                <Card key={prompt.id}>
                  <CardHeader className="pb-3">
                    <div className="flex items-start justify-between">
                      <div className="space-y-1">
                        <CardTitle className="flex items-center gap-2 text-lg">
                          {prompt.name}
                          <Badge variant={prompt.enabled ? 'success' : 'secondary'}>
                            {prompt.enabled ? '启用' : '禁用'}
                          </Badge>
                          <Badge variant="outline">优先级: {prompt.priority}</Badge>
                        </CardTitle>
                        {prompt.description && (
                          <CardDescription>{prompt.description}</CardDescription>
                        )}
                      </div>
                      <div className="flex gap-2">
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => openEditPromptDialog(prompt)}
                        >
                          <Pencil className="h-4 w-4" />
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => setDeletePromptId(prompt.id)}
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="grid gap-3 text-sm">
                      <div className="flex flex-wrap gap-4">
                        {prompt.device_serial && (
                          <div>
                            <span className="text-muted-foreground">设备序列号:</span>
                            <code className="ml-2 bg-muted px-2 py-0.5 rounded text-xs">
                              {prompt.device_serial}
                            </code>
                          </div>
                        )}
                        {prompt.device_model && (
                          <div>
                            <span className="text-muted-foreground">设备型号:</span>
                            <code className="ml-2 bg-muted px-2 py-0.5 rounded text-xs">
                              {prompt.device_model}
                            </code>
                          </div>
                        )}
                        {!prompt.device_serial && !prompt.device_model && (
                          <span className="text-muted-foreground">适用于所有设备</span>
                        )}
                      </div>

                      {prompt.system_prompt && (
                        <div className="p-3 rounded-lg bg-muted/50">
                          <span className="text-muted-foreground text-xs">系统提示词:</span>
                          <p className="mt-1 whitespace-pre-wrap">{prompt.system_prompt}</p>
                        </div>
                      )}
                      {prompt.prefix_prompt && (
                        <div className="p-3 rounded-lg bg-blue-500/10">
                          <span className="text-blue-500 text-xs">前缀指令:</span>
                          <p className="mt-1 whitespace-pre-wrap">{prompt.prefix_prompt}</p>
                        </div>
                      )}
                      {prompt.suffix_prompt && (
                        <div className="p-3 rounded-lg bg-green-500/10">
                          <span className="text-green-500 text-xs">后缀指令:</span>
                          <p className="mt-1 whitespace-pre-wrap">{prompt.suffix_prompt}</p>
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </TabsContent>

        {/* ========== 任务模版 Tab ========== */}
        <TabsContent value="task-templates" className="space-y-6">
          <div className="flex justify-between items-center">
            <p className="text-sm text-muted-foreground">
              创建任务模版，可在创建任务时快速选择或直接创建任务
            </p>
            <Button onClick={openCreateTemplateDialog}>
              <Plus className="h-4 w-4 mr-2" />
              创建模版
            </Button>
          </div>

          {/* 模版列表 */}
          {taskTemplates.length === 0 ? (
            <Card>
              <CardContent className="py-8 text-center">
                <FileCode className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                <p className="text-muted-foreground mb-4">暂无任务模版</p>
                <Button onClick={openCreateTemplateDialog}>创建第一个模版</Button>
              </CardContent>
            </Card>
          ) : (
            <div className="grid gap-4 md:grid-cols-2">
              {taskTemplates.map((template) => {
                const IconComponent = template.icon ? iconMap[template.icon] : FileCode
                return (
                  <Card key={template.id}>
                    <CardHeader className="pb-3">
                      <div className="flex items-start justify-between">
                        <div className="space-y-1">
                          <CardTitle className="flex items-center gap-2 text-lg">
                            {IconComponent && <IconComponent className="h-5 w-5" />}
                            {template.name}
                            {template.category && (
                              <Badge variant="outline">{template.category}</Badge>
                            )}
                          </CardTitle>
                          {template.description && (
                            <CardDescription>{template.description}</CardDescription>
                          )}
                        </div>
                        <div className="flex gap-2">
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => openEditTemplateDialog(template)}
                          >
                            <Pencil className="h-4 w-4" />
                          </Button>
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => setDeleteTemplateId(template.id)}
                          >
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent>
                      <div className="grid gap-3 text-sm">
                        <div className="p-3 rounded-lg bg-muted/50">
                          <span className="text-muted-foreground text-xs">执行指令:</span>
                          <p className="mt-1 whitespace-pre-wrap">{template.command}</p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                )
              })}
            </div>
          )}
        </TabsContent>
      </Tabs>

      {/* ========== 系统提示词 Dialog ========== */}
      <Dialog open={isPromptDialogOpen} onOpenChange={setIsPromptDialogOpen}>
        <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>
              {editingPrompt ? '编辑系统提示词' : '创建系统提示词'}
            </DialogTitle>
            <DialogDescription>
              配置针对特定设备的系统级提示词
            </DialogDescription>
          </DialogHeader>

          {/* 设备预设 */}
          {!editingPrompt && (
            <div className="space-y-2">
              <Label>快速预设</Label>
              <div className="flex flex-wrap gap-2">
                {devicePresets.map((preset, index) => (
                  <Button
                    key={index}
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={() => applyDevicePreset(preset)}
                  >
                    {preset.name}
                  </Button>
                ))}
              </div>
            </div>
          )}

          <form onSubmit={handlePromptSubmit} className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="prompt-name">名称 *</Label>
                <Input
                  id="prompt-name"
                  value={promptFormData.name}
                  onChange={(e) =>
                    setPromptFormData({ ...promptFormData, name: e.target.value })
                  }
                  placeholder="如: 华为手机专用"
                  required
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="prompt-priority">优先级</Label>
                <Input
                  id="prompt-priority"
                  type="number"
                  value={promptFormData.priority}
                  onChange={(e) =>
                    setPromptFormData({ ...promptFormData, priority: parseInt(e.target.value) || 0 })
                  }
                />
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="prompt-description">描述</Label>
              <Input
                id="prompt-description"
                value={promptFormData.description}
                onChange={(e) =>
                  setPromptFormData({ ...promptFormData, description: e.target.value })
                }
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="device_serial">设备序列号 (支持通配符 *)</Label>
                <Input
                  id="device_serial"
                  value={promptFormData.device_serial}
                  onChange={(e) =>
                    setPromptFormData({ ...promptFormData, device_serial: e.target.value })
                  }
                  placeholder="留空匹配所有"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="device_model">设备型号 (支持通配符 *)</Label>
                <Input
                  id="device_model"
                  value={promptFormData.device_model}
                  onChange={(e) =>
                    setPromptFormData({ ...promptFormData, device_model: e.target.value })
                  }
                  placeholder="如: HUAWEI*"
                />
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="system_prompt">系统提示词</Label>
              <Textarea
                id="system_prompt"
                value={promptFormData.system_prompt}
                onChange={(e) =>
                  setPromptFormData({ ...promptFormData, system_prompt: e.target.value })
                }
                rows={3}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="prefix_prompt">前缀指令</Label>
              <Textarea
                id="prefix_prompt"
                value={promptFormData.prefix_prompt}
                onChange={(e) =>
                  setPromptFormData({ ...promptFormData, prefix_prompt: e.target.value })
                }
                rows={3}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="suffix_prompt">后缀指令</Label>
              <Textarea
                id="suffix_prompt"
                value={promptFormData.suffix_prompt}
                onChange={(e) =>
                  setPromptFormData({ ...promptFormData, suffix_prompt: e.target.value })
                }
                rows={3}
              />
            </div>

            <div className="flex items-center justify-between">
              <Label htmlFor="prompt-enabled">启用</Label>
              <Switch
                id="prompt-enabled"
                checked={promptFormData.enabled}
                onCheckedChange={(checked) =>
                  setPromptFormData({ ...promptFormData, enabled: checked })
                }
              />
            </div>

            <DialogFooter>
              <Button type="button" variant="outline" onClick={closePromptDialog}>
                取消
              </Button>
              <Button
                type="submit"
                disabled={createPromptMutation.isPending || updatePromptMutation.isPending}
              >
                {editingPrompt ? '保存' : '创建'}
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>

      {/* ========== 任务模版 Dialog ========== */}
      <Dialog open={isTemplateDialogOpen} onOpenChange={setIsTemplateDialogOpen}>
        <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>
              {editingTemplate ? '编辑任务模版' : '创建任务模版'}
            </DialogTitle>
            <DialogDescription>
              创建可复用的任务模版
            </DialogDescription>
          </DialogHeader>

          {/* 预设模版 */}
          {!editingTemplate && (
            <div className="space-y-2">
              <Label>快速预设</Label>
              <div className="flex flex-wrap gap-2">
                {presetTemplates.map((preset, index) => (
                  <Button
                    key={index}
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={() => applyPresetTemplate(preset)}
                  >
                    {preset.name}
                  </Button>
                ))}
              </div>
            </div>
          )}

          <form onSubmit={handleTemplateSubmit} className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="template-name">模版名称 *</Label>
                <Input
                  id="template-name"
                  value={templateFormData.name}
                  onChange={(e) =>
                    setTemplateFormData({ ...templateFormData, name: e.target.value })
                  }
                  placeholder="如: 蚂蚁森林收能量"
                  required
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="template-category">分类</Label>
                <Input
                  id="template-category"
                  value={templateFormData.category}
                  onChange={(e) =>
                    setTemplateFormData({ ...templateFormData, category: e.target.value })
                  }
                  placeholder="如: 日常任务"
                />
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="template-description">描述</Label>
              <Input
                id="template-description"
                value={templateFormData.description}
                onChange={(e) =>
                  setTemplateFormData({ ...templateFormData, description: e.target.value })
                }
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="template-command">执行指令 *</Label>
              <Textarea
                id="template-command"
                value={templateFormData.command}
                onChange={(e) =>
                  setTemplateFormData({ ...templateFormData, command: e.target.value })
                }
                placeholder="输入要执行的自然语言指令..."
                rows={4}
                required
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="template-icon">图标</Label>
              <Input
                id="template-icon"
                value={templateFormData.icon}
                onChange={(e) =>
                  setTemplateFormData({ ...templateFormData, icon: e.target.value })
                }
                placeholder="如: tree-deciduous"
              />
              <p className="text-xs text-muted-foreground">
                可选: tree-deciduous, coffee, gift, zap
              </p>
            </div>

            <DialogFooter>
              <Button type="button" variant="outline" onClick={closeTemplateDialog}>
                取消
              </Button>
              <Button
                type="submit"
                disabled={createTemplateMutation.isPending || updateTemplateMutation.isPending}
              >
                {editingTemplate ? '保存' : '创建'}
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>

      {/* 删除提示词确认对话框 */}
      <AlertDialog open={deletePromptId !== null} onOpenChange={(open) => !open && setDeletePromptId(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>确认删除</AlertDialogTitle>
            <AlertDialogDescription>
              确定要删除这个提示词吗？此操作不可撤销。
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>取消</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => {
                if (deletePromptId !== null) {
                  deletePromptMutation.mutate(deletePromptId)
                  setDeletePromptId(null)
                }
              }}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              删除
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* 删除模版确认对话框 */}
      <AlertDialog open={deleteTemplateId !== null} onOpenChange={(open) => !open && setDeleteTemplateId(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>确认删除</AlertDialogTitle>
            <AlertDialogDescription>
              确定要删除这个模版吗？此操作不可撤销。
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>取消</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => {
                if (deleteTemplateId !== null) {
                  deleteTemplateMutation.mutate(deleteTemplateId)
                  setDeleteTemplateId(null)
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
