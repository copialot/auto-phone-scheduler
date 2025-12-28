import { useState } from 'react'
import { toast } from 'sonner'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { Badge } from '@/components/ui/badge'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
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
import { notificationsApi } from '@/api/client'
import type { NotificationChannel, NotificationChannelCreate, NotificationType } from '@/types'
import { Plus, Pencil, Trash2, Send, MessageSquare } from 'lucide-react'

export function Notifications() {
  const queryClient = useQueryClient()
  const [isDialogOpen, setIsDialogOpen] = useState(false)
  const [editingChannel, setEditingChannel] = useState<NotificationChannel | null>(null)
  const [channelType, setChannelType] = useState<NotificationType>('dingtalk')
  const [deleteChannelId, setDeleteChannelId] = useState<number | null>(null)
  const [formData, setFormData] = useState({
    name: '',
    enabled: true,
    // DingTalk
    webhook: '',
    secret: '',
    // Telegram
    bot_token: '',
    chat_id: '',
  })

  const { data: channels = [], isLoading } = useQuery({
    queryKey: ['notifications'],
    queryFn: notificationsApi.list,
  })

  const createMutation = useMutation({
    mutationFn: notificationsApi.create,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['notifications'] })
      closeDialog()
    },
  })

  const updateMutation = useMutation({
    mutationFn: ({ id, data }: { id: number; data: Partial<NotificationChannelCreate> }) =>
      notificationsApi.update(id, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['notifications'] })
      closeDialog()
    },
  })

  const deleteMutation = useMutation({
    mutationFn: notificationsApi.delete,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['notifications'] })
    },
  })

  const testMutation = useMutation({
    mutationFn: notificationsApi.test,
    onSuccess: () => {
      toast.success('测试通知发送成功')
    },
    onError: () => {
      toast.error('测试通知发送失败，请检查配置')
    },
  })

  const openCreateDialog = (type: NotificationType) => {
    setEditingChannel(null)
    setChannelType(type)
    setFormData({
      name: type === 'dingtalk' ? '钉钉通知' : 'Telegram通知',
      enabled: true,
      webhook: '',
      secret: '',
      bot_token: '',
      chat_id: '',
    })
    setIsDialogOpen(true)
  }

  const openEditDialog = (channel: NotificationChannel) => {
    setEditingChannel(channel)
    setChannelType(channel.type)
    setFormData({
      name: channel.name,
      enabled: channel.enabled,
      webhook: channel.config.webhook || '',
      secret: channel.config.secret || '',
      bot_token: channel.config.bot_token || '',
      chat_id: channel.config.chat_id || '',
    })
    setIsDialogOpen(true)
  }

  const closeDialog = () => {
    setIsDialogOpen(false)
    setEditingChannel(null)
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    const config: Record<string, string> =
      channelType === 'dingtalk'
        ? { webhook: formData.webhook, secret: formData.secret }
        : { bot_token: formData.bot_token, chat_id: formData.chat_id }

    const data: NotificationChannelCreate = {
      type: channelType,
      name: formData.name,
      config,
      enabled: formData.enabled,
    }

    if (editingChannel) {
      updateMutation.mutate({ id: editingChannel.id, data })
    } else {
      createMutation.mutate(data)
    }
  }

  const getTypeIcon = (_type: string) => {
    return <MessageSquare className="h-5 w-5" />
  }

  if (isLoading) {
    return <div>加载中...</div>
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold">通知配置</h1>
      </div>

      {/* Add Channel Cards */}
      <div className="grid gap-4 md:grid-cols-2">
        <Card
          className="cursor-pointer hover:bg-muted/50 transition-colors"
          onClick={() => openCreateDialog('dingtalk')}
        >
          <CardContent className="py-6 flex items-center gap-4">
            <div className="p-3 rounded-lg bg-blue-500/10">
              <MessageSquare className="h-6 w-6 text-blue-500" />
            </div>
            <div>
              <h3 className="font-medium">钉钉机器人</h3>
              <p className="text-sm text-muted-foreground">
                通过钉钉群机器人接收通知
              </p>
            </div>
            <Plus className="h-5 w-5 ml-auto text-muted-foreground" />
          </CardContent>
        </Card>

        <Card
          className="cursor-pointer hover:bg-muted/50 transition-colors"
          onClick={() => openCreateDialog('telegram')}
        >
          <CardContent className="py-6 flex items-center gap-4">
            <div className="p-3 rounded-lg bg-sky-500/10">
              <MessageSquare className="h-6 w-6 text-sky-500" />
            </div>
            <div>
              <h3 className="font-medium">Telegram Bot</h3>
              <p className="text-sm text-muted-foreground">
                通过 Telegram Bot 接收通知
              </p>
            </div>
            <Plus className="h-5 w-5 ml-auto text-muted-foreground" />
          </CardContent>
        </Card>
      </div>

      {/* Existing Channels */}
      <div className="space-y-4">
        <h2 className="text-xl font-semibold">已配置渠道</h2>
        {channels.length === 0 ? (
          <Card>
            <CardContent className="py-8 text-center">
              <p className="text-muted-foreground">暂无通知渠道，点击上方添加</p>
            </CardContent>
          </Card>
        ) : (
          <div className="grid gap-4">
            {channels.map((channel) => (
              <Card key={channel.id}>
                <CardContent className="py-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      {getTypeIcon(channel.type)}
                      <div>
                        <div className="flex items-center gap-2">
                          <span className="font-medium">{channel.name}</span>
                          <Badge variant={channel.enabled ? 'success' : 'secondary'}>
                            {channel.enabled ? '启用' : '禁用'}
                          </Badge>
                          <Badge variant="outline">
                            {channel.type === 'dingtalk' ? '钉钉' : 'Telegram'}
                          </Badge>
                        </div>
                      </div>
                    </div>
                    <div className="flex gap-2">
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => testMutation.mutate(channel.id)}
                        disabled={testMutation.isPending}
                      >
                        <Send className="h-4 w-4" />
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => openEditDialog(channel)}
                      >
                        <Pencil className="h-4 w-4" />
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => setDeleteChannelId(channel.id)}
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
      </div>

      {/* Create/Edit Dialog */}
      <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>
              {editingChannel ? '编辑通知渠道' : '添加通知渠道'}
            </DialogTitle>
          </DialogHeader>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="name">渠道名称</Label>
              <Input
                id="name"
                value={formData.name}
                onChange={(e) =>
                  setFormData({ ...formData, name: e.target.value })
                }
                required
              />
            </div>

            {channelType === 'dingtalk' ? (
              <>
                <div className="space-y-2">
                  <Label htmlFor="webhook">Webhook URL</Label>
                  <Input
                    id="webhook"
                    value={formData.webhook}
                    onChange={(e) =>
                      setFormData({ ...formData, webhook: e.target.value })
                    }
                    placeholder="https://oapi.dingtalk.com/robot/send?access_token=xxx"
                    required
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="secret">签名密钥 (可选)</Label>
                  <Input
                    id="secret"
                    value={formData.secret}
                    onChange={(e) =>
                      setFormData({ ...formData, secret: e.target.value })
                    }
                    placeholder="SEC..."
                  />
                </div>
              </>
            ) : (
              <>
                <div className="space-y-2">
                  <Label htmlFor="bot_token">Bot Token</Label>
                  <Input
                    id="bot_token"
                    value={formData.bot_token}
                    onChange={(e) =>
                      setFormData({ ...formData, bot_token: e.target.value })
                    }
                    placeholder="123456:ABC-DEF..."
                    required
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="chat_id">Chat ID</Label>
                  <Input
                    id="chat_id"
                    value={formData.chat_id}
                    onChange={(e) =>
                      setFormData({ ...formData, chat_id: e.target.value })
                    }
                    placeholder="-1001234567890"
                    required
                  />
                </div>
              </>
            )}

            <div className="flex items-center justify-between">
              <Label htmlFor="enabled">启用</Label>
              <Switch
                id="enabled"
                checked={formData.enabled}
                onCheckedChange={(checked) =>
                  setFormData({ ...formData, enabled: checked })
                }
              />
            </div>

            <DialogFooter>
              <Button type="button" variant="outline" onClick={closeDialog}>
                取消
              </Button>
              <Button
                type="submit"
                disabled={createMutation.isPending || updateMutation.isPending}
              >
                {editingChannel ? '保存' : '添加'}
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>

      {/* 删除通知渠道确认对话框 */}
      <AlertDialog open={deleteChannelId !== null} onOpenChange={(open) => !open && setDeleteChannelId(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>确认删除</AlertDialogTitle>
            <AlertDialogDescription>
              确定要删除这个通知渠道吗？此操作不可撤销。
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>取消</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => {
                if (deleteChannelId !== null) {
                  deleteMutation.mutate(deleteChannelId)
                  setDeleteChannelId(null)
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
