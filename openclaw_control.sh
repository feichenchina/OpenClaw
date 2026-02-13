#!/bin/bash
# OpenClaw 网关管理脚本 - 适用于 Systemd 用户服务模式
# 使用方法: ./openclaw_control.sh {start|stop|restart|status|log|enable|disable}

SERVICE="openclaw-gateway.service"
LOG_FILE="/tmp/openclaw/gateway.log"

# 检查 systemd 用户服务是否存在
if ! systemctl --user list-unit-files | grep -q "$SERVICE"; then
    echo "❌ 错误: 找不到服务 $SERVICE"
    echo "   请确认服务文件存在于 ~/.config/systemd/user/ 目录"
    exit 1
fi

case "$1" in
    start)
        echo "🚀 启动 OpenClaw 网关..."
        systemctl --user start "$SERVICE"
        sleep 1
        systemctl --user status "$SERVICE" --no-pager
        ;;
    stop)
        echo "🛑 停止 OpenClaw 网关..."
        systemctl --user stop "$SERVICE"
        ;;
    restart)
        echo "🔄 重启 OpenClaw 网关..."
        systemctl --user restart "$SERVICE"
        sleep 2
        systemctl --user status "$SERVICE" --no-pager
        ;;
    status)
        echo "📊 OpenClaw 网关状态:"
        systemctl --user status "$SERVICE" --no-pager
        ;;
    log)
        if [ -f "$LOG_FILE" ]; then
            echo "📋 实时日志 (按 Ctrl+C 退出):"
            tail -f "$LOG_FILE"
        else
            echo "⚠️  日志文件不存在: $LOG_FILE"
            echo "   请确认服务已启动并配置了 StandardOutput/Error"
        fi
        ;;
    enable)
        echo "🔌 设置 OpenClaw 开机自启..."
        systemctl --user enable "$SERVICE"
        ;;
    disable)
        echo "🔌 取消 OpenClaw 开机自启..."
        systemctl --user disable "$SERVICE"
        ;;
    *)
        echo "📌 用法: $0 {start|stop|restart|status|log|enable|disable}"
        echo ""
        echo "示例:"
        echo "  $0 start    - 启动服务"
        echo "  $0 status   - 查看状态"
        echo "  $0 log      - 滚动查看日志"
        exit 1
        ;;
esac
