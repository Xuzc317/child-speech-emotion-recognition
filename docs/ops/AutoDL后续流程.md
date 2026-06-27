# AutoDL 后续流程

> 更新: 2026-06-16 | 192 runs 全部完成

## 当前状态

- 所有实验已完成，数据全量同步到本地
- AutoDL 实例可关机，无待跑任务
- SSH: `connect.cqa1.seetacloud.com:25808`

## 同步数据

```bash
# 拉取全部结果 JSON
python scripts/tmp_paramiko_autodl_runner.py --pull-all

# 下载云端权重
python scripts/download_checkpoints.py

# 校验完整性
python scripts/verify_all_192.py
```

## 开机后操作

1. 在 AutoDL 控制台开机（需 GPU 实例）
2. 等待实例就绪（约 2 分钟）
3. 如需重跑实验：`bash scripts/launch_b*.sh`
4. 同步结果：`python scripts/tmp_paramiko_autodl_runner.py --pull-all`

## 关机

在 AutoDL 控制台直接关机即可。所有数据已本地备份。
