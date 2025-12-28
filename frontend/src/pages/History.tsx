import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
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
import { executionsApi } from '@/api/client'
import { formatDateTime } from '@/lib/utils'
import { Eye, Trash2 } from 'lucide-react'

export function History() {
  const queryClient = useQueryClient()
  const navigate = useNavigate()
  const [deleteId, setDeleteId] = useState<number | null>(null)

  const { data: executions = [], isLoading } = useQuery({
    queryKey: ['executions'],
    queryFn: () => executionsApi.list(),
  })

  const deleteMutation = useMutation({
    mutationFn: executionsApi.delete,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['executions'] })
      setDeleteId(null)
    },
  })

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
    return <div>加载中...</div>
  }

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold">执行历史</h1>

      {executions.length === 0 ? (
        <Card>
          <CardContent className="py-8 text-center">
            <p className="text-muted-foreground">暂无执行记录</p>
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-4">
          {executions.map((exec) => (
            <Card key={exec.id}>
              <CardContent className="py-4">
                <div className="flex items-center justify-between">
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <span className="font-medium">
                        {exec.task_name || `任务 #${exec.task_id}`}
                      </span>
                      {getStatusBadge(exec.status)}
                    </div>
                    <div className="text-sm text-muted-foreground">
                      {exec.started_at && (
                        <span>
                          开始: {formatDateTime(exec.started_at)}
                        </span>
                      )}
                      {exec.finished_at && (
                        <span className="ml-4">
                          结束: {formatDateTime(exec.finished_at)}
                        </span>
                      )}
                    </div>
                    {exec.error_message && (
                      <p className="text-sm text-destructive">
                        错误: {exec.error_message}
                      </p>
                    )}
                  </div>
                  <div className="flex gap-2">
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => navigate(`/history/${exec.id}`)}
                    >
                      <Eye className="h-4 w-4" />
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => setDeleteId(exec.id)}
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

      {/* 删除确认对话框 */}
      <AlertDialog open={deleteId !== null} onOpenChange={(open) => !open && setDeleteId(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>确认删除</AlertDialogTitle>
            <AlertDialogDescription>
              确定要删除这条执行记录吗？此操作无法撤销。
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>取消</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => deleteId && deleteMutation.mutate(deleteId)}
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
