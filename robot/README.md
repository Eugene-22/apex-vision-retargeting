# 右手 ApexHand 控制

控制层连接现有 `perception → retargeting` 输出与 `vendor/rysen-sdk/python/rysen_apexhand_sdk.py`。只接受右手目标，连接后使用 `get_hand_dir() == HandDir.RIGHT` 核对设备；左手或未知结果会断开，且不使能。

## 运行

在项目根目录的 Python 3.10 环境中：

```bash
# 感知、重定向、控制节拍与限速空跑；不加载 SDK，不连接设备
python -m robot --camera 0 --show

# 回放 perception 保存的 JSONL，用于空跑控制层
python -m robot --replay landmarks.jsonl

# 真机右手：将地址替换为实际设备 IP
python -m robot --execute --ip YOUR_RIGHT_HAND_IP --camera 0 --show

# 已镜像的相机源
python -m robot --execute --ip YOUR_RIGHT_HAND_IP --camera 0 --input-mirrored
```

`q`、Esc 或达到 `--max-frames` 正常结束时，已开始跟随且无故障的会话会先停止跟随线程，按官方 example 的 `move_joint` 流程将 21 个关节低速移到 0 弧度，检查新反馈后，再禁用手指、注销回调和断开。跟随期间按 Ctrl+C、跟踪超时、硬件故障或其他异常会直接禁用并断开；尚未开始跟随的会话也不会回零或额外使能。没有预设设备 IP，不会自动扫描网络或修改 IP、清除故障。记录回放仅用于空跑，按原记录时间间隔运行；真机入口使用实时相机。

正常退出时会显示 `Returning RIGHT hand to zero pose before disconnect...`，反馈确认后显示 `Zero pose confirmed; disabling fingers.`。回零速度最多 0.5 rad/s、加速度最多 2 rad/s²，并遵守配置中更低的限值。回零结束需新反馈中的全部 21 个关节距零位不超过 2°；反馈过期、回零失败或无法确认时仍尝试禁用和断开，并报告错误。`move_joint` 是 SDK 阻塞接口；随后最多 2 秒的反馈确认等待不代表能限制原生回零调用本身的耗时。

若启动前已经越限，仍需先使用已验证的官方流程恢复起始姿态；本程序不会自动绕过启动限位。后续请在跟随正常时按 `q` 或 Esc 退出，观察回零完成后再重新启动，验证禁用后的位置是否仍符合限位。

`--record landmarks.jsonl` 可录制输入关键点；标准输出为每帧 JSON，包含目标角、最近下发角和指令计数。指令线程异步运行，因此首帧可能显示指令计数为零。`--max-frames N` 可用于有限帧验证。

配置在 `config/robot_right.yaml`：

| 参数 | 默认值 | 用途 |
| --- | --- | --- |
| `hand_side` | right | 固定为右手，其他值报错 |
| `ip` | null | 真机必须通过配置或 `--ip` 提供 |
| `command_rate` | 30 Hz | 独立控制线程的目标频率上限 |
| `max_speed` | 1 rad/s | 软件逐帧限速与 SDK 速度限制，另受 URDF 限制 |
| `max_accel` | 5 rad/s² | SDK 位置跟随规划器的加速度限制 |
| `finger_torque_pct` | 30% | SDK 五指扭矩上限参数 |
| `target_timeout` | 0.5 s | 有效目标超时后停止并禁用手指 |
| `feedback_timeout` | 0.5 s | 反馈回调过期后停止并禁用手指 |
| `startup_timeout` | 3 s | 等待完整实测关节反馈的最长时间 |
| `feedback_rate` | 100 Hz | SDK 关节反馈回调频率 |
| `return_to_neutral_on_exit` | true | 正常退出前回到零位；设 false 可关闭 |

配置路径相对 YAML 文件解析，真机控制与重定向必须使用同一个右手 URDF。扭矩百分比是 SDK 参数，不能替代实际指尖接触力标定。

## SDK 与目标角接口

```python
from robot import RobotConfig, RightHandController

# qpos21 来自 Retargeter(..., hand_side="right")
with RightHandController(RobotConfig(ip="YOUR_RIGHT_HAND_IP")) as hand:
    hand.submit(qpos21, hand_side="right")
    # 后续持续提交新目标；离开 with 会关闭设备。
```

初始化和导入不会发出硬件命令。进入上下文后连接、加载右手模型、设置参数并等待反馈；首次有效 `submit` 才使能。控制线程从实测角度开始逐步接近目标，不使用全零角度替代缺失反馈。

- `submit(qpos21)`：21 个弧度角，顺序为拇指 0–4、食指 5–8、中指 9–12、无名指 13–16、小指 17–20。要求五组联动一致；拒绝 NaN、错误长度和左手目标。
- `submit_actuated(qpos16)`：兼容参考 `landmarks_to_actuated` 的 16 维输出，顺序为拇指 4 项，其余四指各 3 项。按 URDF 展开为 21 维，并统一限位。小指逻辑名称 `pinky` 对应本地 SDK 的 `LITTLE`。
- `observed_at`：可传本机 `time.monotonic()` 的采集时间，让模型推理和优化耗时也计入目标有效期。不能传录像时间戳或 Unix 时间戳。省略时使用提交时间。
- `check_health()`：向主循环报告控制线程的故障；超时或 SDK 错误会锁定停止状态，不会自动重新使能，需检查后重新创建控制器。
- `last_command`：最近的完整 21 维输出副本；`command_count` 是发送成功次数。

下发调用为 `move_j_position_follow`。本地 SDK 的 `MoveJPositionFollowParam` 使用 **`id` 和 `position`**，不带上游旧版本示例中的 `torque_nmm`。扭矩上限通过 `MaxFingerTorque` 单独配置。所有返回值均与 `ErrorCode.ERROR_CODE_OK` 比较，不能用布尔真值判断，因为成功码是零。

参考文件使用旧 `mp.solutions` 与图像坐标，现有感知模块使用 Tasks API 和米制 world landmarks；控制层接收的是已经重定向的关节角，不再进行图像缩放、左右镜像或外展角反转。参考映射的外展修正不能再次应用到当前 URDF 优化结果。关节限位以本地 URDF 为准，例如本地拇指 joint0 上限约 80°，不是参考关节表中的 90°。

## 丢失、故障与关闭

控制线程只保存最新目标，不积压旧帧。短暂丢失时最多在 `target_timeout` 内继续接近最后一个有效目标；超过期限禁用手指并锁定停止，即使感知循环阻塞，也由独立线程检查超时。每次发送前检查连接、硬件错误码与最近反馈。

缺失、不完整、重复 ID 或含非有限数值的反馈不能用于启动。发送失败不会更新最近指令。清理过程中即使禁用失败，仍尝试注销回调和断开，并报告失败。Python 线程和 SDK 网络调用不是硬实时系统；若跟随线程中的 SDK 调用阻塞超过线程关闭等待时间，会明确报告无法确认软件停止，不宣称机器人已停止。

## SDK 退出崩溃修复

当前相机仍使用同步采集与识别。针对 SDK 1.5.2 注销关节反馈回调时的崩溃，适配层保留回调对象引用，直到注销、断开及 SDK 对象释放完成后再释放。初始化或主循环出错时，会在清理前输出原始错误，避免清理崩溃掩盖退出原因。

若报 `Measured startup position lies outside right-hand URDF limits`，错误会列出越限主动关节的名称、SDK ID、实测角度及允许范围（度），供核对设备零位、标定和右手模型。原有关节限位与启动保护保留，越限时不会使能。

修复的离线回归覆盖正常关闭、注册失败清理、原生库释放回调引用、清理前输出原始异常，以及启动越限时不使能；不连接实体手。含真实 MediaPipe 模型集成的完整测试为 **74 项通过**。真实 SDK 的退出行为仍需现场复测。

## 运行环境与验证范围

厂商当前二进制面向 Linux x86_64/aarch64、CPython 3.10。模块直接按路径加载 vendor 中的 Python 包装器，无需另装不同版本 SDK。原生运行库可按厂商 `vendor/rysen-sdk/install_rysen_deps.sh` 安装，或使用现有 `rysen_sdk:latest` 容器；相机设备与显示映射需要在容器环境中另外配置。

首次实现时宿主机缺少部分原生依赖，使用替身验证 SDK 接口。后续用户现场日志已确认 SDK 1.5.2 能加载并连接设备；自动化测试仍不连接或驱动实体手。空跑模式不依赖原生 SDK。

```bash
python -m pytest -q tests/test_robot.py
```

测试包含：替身原生扩展下执行**真实 vendor Python 包装器**、右手判定、16→21 关节展开、SDK 参数类型/顺序、测量姿态启动、限位和限速、过期目标/反馈、独立看门狗、初始化/发送/关闭失败、无 SDK 空跑以及回放命令行端到端流程。真实相机到实体右手的运动效果与零位仍待现场联调。

2026-09-06：31 项 robot 测试通过；加上感知、重定向与真实 MediaPipe 模型集成测试，完整回归共 **69 项通过**。

正常退出回零更新后，控制层 **52 项测试通过**，完整回归 **90 项通过**。覆盖真实 Python SDK 包装器的 MoveJoint 参数、新反馈确认、回零失败清理、退出顺序及故障/Ctrl+C 不回零。未驱动实体手，连续启动的效果仍需现场验证。

## 参考

- [hand_tracker.py](https://github.com/peterpanstechland/apexhand/blob/main/tracking/hand_tracker.py)
- [retarget.py](https://github.com/peterpanstechland/apexhand/blob/main/tracking/retarget.py)
- [参考 SDK 接口](https://github.com/peterpanstechland/apexhand/blob/main/real/apex_interface.py)

采用其“跟踪结果 → 主动关节 → SDK 补齐联动”的接口思路，依据本地 SDK 独立实现控制层；现有 perception/retargeting 算法保持原接口。
