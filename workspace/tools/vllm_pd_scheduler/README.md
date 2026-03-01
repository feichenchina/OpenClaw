# vLLM PD Disaggregation Dynamic Scheduler

基于 vLLM 的 Prefill-Decode (PD) 分离动态调度模块。

## 概述

在 vLLM 的 PD 分离架构中，推理过程被拆分为两个阶段：

- **Prefill (P)**: 处理完整的 prompt 输入，计算密集型
- **Decode (D)**: 逐 token 生成输出，显存带宽密集型

本模块实现了一个动态调度器，将请求智能路由到对应的 P/D 实例，支持多种调度策略和实例健康监控。

## 架构

```
┌───────────────┐     ┌─────────────┐     ┌──────────────┐
│  Client       │────▶│  PDRouter   │────▶│  P-Instance  │
│  Request      │     │  (调度入口)  │     │  (Prefill)   │
└───────────────┘     │             │     └──────┬───────┘
                      │ PDScheduler │            │ KV Cache
                      │ HealthCheck │            │ Transfer
                      │             │     ┌──────▼───────┐
                      │             │────▶│  D-Instance  │
                      └─────────────┘     │  (Decode)    │
                                          └──────────────┘
```

## 调度策略

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| `round_robin` | 轮询分配 | 实例规格一致，负载均匀 |
| `least_load` | 最少等待请求优先 | 通用场景（默认） |
| `weighted` | 综合负载/KV缓存/延迟加权评分 | 异构集群或精细调优 |

## 使用方法

### 1. 配置

创建 JSON 配置文件（参考 `scheduler_config.example.json`）：

```json
{
  "strategy": "least_load",
  "health_check_interval": 5.0,
  "prefill_instances": [
    {"id": "prefill-0", "host": "10.0.0.1", "port": 8000}
  ],
  "decode_instances": [
    {"id": "decode-0", "host": "10.0.0.2", "port": 8000}
  ]
}
```

### 2. 启动调度器

```bash
python -m vllm_pd_scheduler --config scheduler_config.json
```

可选参数：

- `--interval <秒>`: 覆盖健康检查间隔
- `--log-level DEBUG|INFO|WARNING`: 日志级别

### 3. 编程接口

```python
from vllm_pd_scheduler.config import load_config
from vllm_pd_scheduler.models import InstanceRole, PDInstance
from vllm_pd_scheduler.router import PDRouter

config = load_config("scheduler_config.json")
router = PDRouter(config)

# 注册实例
router.register_instance(PDInstance(
    instance_id="p1", role=InstanceRole.PREFILL,
    host="10.0.0.1", port=8000
))
router.register_instance(PDInstance(
    instance_id="d1", role=InstanceRole.DECODE,
    host="10.0.0.2", port=8000
))

# 健康检查
router.run_health_checks()

# 路由请求（自动经过 prefill → decode 流水线）
result = router.route_request({"prompt": "你好", "max_tokens": 100})
print(result.success, result.instance.instance_id)
```

### 4. 动态管理实例

```python
# 排空实例（不再接收新请求）
router.drain_instance("p1")

# 移除实例
router.unregister_instance("p1")

# 查看特定角色的实例
prefills = router.get_instances(role=InstanceRole.PREFILL)
```

## 运行测试

```bash
cd workspace/tools
python -m pytest vllm_pd_scheduler/tests/ -v
# 或者
python -m unittest vllm_pd_scheduler.tests.test_scheduler -v
```

## 模块结构

```
vllm_pd_scheduler/
├── __init__.py          # 包入口
├── __main__.py          # CLI 入口
├── config.py            # 配置加载
├── health.py            # 实例健康检查
├── models.py            # 数据模型
├── router.py            # 请求路由（核心入口）
├── scheduler.py         # 调度策略实现
├── scheduler_config.example.json  # 示例配置
└── tests/
    └── test_scheduler.py  # 单元测试
```

## 配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `strategy` | string | `least_load` | 调度策略 |
| `health_check_interval` | float | 5.0 | 健康检查间隔（秒） |
| `health_check_timeout` | float | 2.0 | 健康检查超时（秒） |
| `max_retries` | int | 3 | 最大重试次数 |
| `drain_timeout` | float | 30.0 | 排空超时（秒） |
| `load_weight` | float | 0.5 | 加权策略中负载权重 |
| `kv_cache_weight` | float | 0.3 | 加权策略中 KV 缓存权重 |
| `latency_weight` | float | 0.2 | 加权策略中延迟权重 |
