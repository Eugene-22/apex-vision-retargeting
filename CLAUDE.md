# CLAUDE.md — Apex Vision Retargeting

## 0. Project Mission

本项目用于完成“基于视觉姿态捕捉的灵巧手姿态重定向”任务。

最终系统目标：

Camera
→ RGB Frame
→ Human Hand Pose Estimation
→ Human Hand Representation
→ Retargeting
→ Apex Hand Joint Targets
→ Safety / Filtering
→ Rysen SDK / ROS2
→ Apex Hand

核心评价指标：

1. 人手姿态还原准确度
2. 灵巧手动作同步一致性
3. 视觉感知稳定性
4. 姿态解算精度
5. 实时性与端到端延迟
6. 机器人控制稳定性
7. 遮挡、快速运动等情况下的鲁棒性

本项目所有代码必须按照可测试、可复现、可扩展的软件工程方式设计，最终需要能够直接接入真实 Apex Hand。

# 1. Repository Boundaries

当前工作目录：

~/projects/
├── rysen-sdk/                  # 官方硬件 SDK，第三方代码
├── apex-hand-urdf/             # 官方机器人 URDF，第三方代码
└── apex-vision-retargeting/    # 本项目，所有自主开发代码


只允许主动修改：apex-vision-retargeting/


默认禁止修改：

../rysen-sdk/

../apex-hand-urdf/



# 2. Runtime Environment

开发环境：

```text
Windows
↓
WSL2 Ubuntu 24.04
↓
Docker / VS Code Dev Container
↓
Rysen SDK based development environment
```

主要 runtime：

```text
Python 3.10
ROS2 Humble
OpenCV
MediaPipe
NumPy
SciPy
pytest
Rysen ApexHand SDK
```


Docker 内项目路径：

```text
/workspace/apex-vision-retargeting
```

Vendor mount：

```text
/workspace/rysen-sdk
/workspace/apex-hand-urdf
```

开发时优先运行：

```bash
python3 -m pytest
```

而不是依赖裸 `pytest` CLI，因为部分容器环境中 pytest entrypoint 不在 PATH。

---

# 3. Current Project Structure

目标结构如下：

```text
apex-vision-retargeting/
├── .devcontainer/
│   └── devcontainer.json
│
├── perception/
│   ├── __init__.py
│   └── mediapipe_hand.py
│
├── human_hand/
│   ├── __init__.py
│   ├── palm_frame.py
│   └── joint_angles.py
│
├── retargeting/
│   ├── __init__.py
│   └── ...
│
├── robot/
│   ├── __init__.py
│   └── ...
│
├── filtering/
│   ├── __init__.py
│   └── ...
│
├── visualization/
│   ├── __init__.py
│   └── hand_draw.py
│
├── scripts/
│   ├── check_env.py
│   ├── test_camera.py
│   └── run_hand_landmarks.py
│
├── tests/
│   ├── test_palm_frame.py
│   └── test_joint_angles.py
│
├── models/
│   └── hand_landmarker.task
│
├── outputs/
│
├── Dockerfile
├── docker-compose.yml
└── CLAUDE.md
```

如果实际目录与这里不同：

1. 先检查现有代码。
2. 不要无理由重构。
3. 先报告差异。
4. 在保持已有功能的情况下逐步整理。

---

# 4. Engineering Rules

## 4.1 每个 Milestone 必须遵循以下顺序

任何新模块都必须按照：

```text
数学/接口定义
→ 最小实现
→ 单元测试
→ synthetic test
→ 真实数据测试
→ 指标记录
→ 验收
```

禁止：

```text
直接一口气实现整条 pipeline
```

禁止同时修改多个尚未验证的层。

例如：

```text
视觉错误
+
角度错误
+
retargeting 错误
+
机器人错误
```

不能同时调试。

必须逐层 isolate。

---

## 4.2 所有数学代码必须有明确坐标约定

任何涉及：

```text
位置
方向
旋转
角度
FK
IK
Jacobian
```

的代码必须明确说明：

```text
Frame
Axis
Units
Angle sign
Rotation convention
```

项目内部统一：

```text
distance: meter 或归一化无量纲
angle: radian
angular velocity: rad/s
time: second
timestamp: monotonic clock
```

只有日志/UI 可以转换为 degree。

禁止算法内部到处混用：

```text
degree
radian
```

---

## 4.3 不允许使用 Magic Number

例如禁止：

```python
angle = angle * 1.37 + 0.23
```

如果确实需要：

```text
offset
scale
joint range
filter coefficient
threshold
```

必须进入：

```text
config / calibration / dataclass
```

并说明来源。

---

## 4.4 第三方数据不能盲信

MediaPipe 的：

```python
hand_landmarks
```

与：

```python
hand_world_landmarks
```

必须区分。

原则：

```text
hand_landmarks
→ 图像可视化

hand_world_landmarks
→ 人手相对 3D 几何计算
```

MediaPipe world landmarks 是模型估计结果，不是真实 RGB-D 深度测量。

任何算法设计不得把它宣称为真实传感器深度。

---

# 5. Hardware Model

Apex Hand：

```text
21 mechanical DoF
16 active DoF
5 coupled/passive DoF
```

四个普通手指每根：

```text
j0 MCP abduction/adduction
j1 MCP flexion
j2 PIP flexion
j3 DIP flexion
```

其中：

```text
j3 mimic j2
ratio = 1:1
```

因此每根普通手指只有：

```text
3 active DoF
```

拇指：

```text
thumb_j0
thumb_j1
thumb_j2
thumb_j3
thumb_j4
```

其中：

```text
thumb_j4 mimic thumb_j3
```

因此：

```text
4 active thumb DoF
```

总计：

```text
4 + 4 × 3 = 16 active DoF
```

禁止向被动 mimic joint 单独发送控制命令。

---

# 6. Apex Joint Limits

从官方 URDF/文档读取并再次校验，不要盲目重复硬编码。

当前已知逻辑范围：

```text
thumb_j0    0°   ~ 90°
thumb_j1   -10°  ~ 60°
thumb_j2    0°   ~ 80°
thumb_j3   -20°  ~ 80°
thumb_j4   mimic thumb_j3

index_j0   -25°  ~ 25°
index_j1   -20°  ~ 90°
index_j2   -5°   ~ 100°
index_j3   mimic index_j2

middle/ring/pinky 同类结构
```

实际代码优先从：

```text
/workspace/apex-hand-urdf
```

解析 limit 和 mimic relationship。

不要让两套独立 hard-coded URDF 参数长期存在。

---

# 7. Completed Work

以下工作已经基本完成。

在继续工作之前必须检查当前仓库确认实际状态，不允许假设所有文件内容完全与此文档一致。

---

## M0 — Development Environment

已完成：

```text
Windows + WSL2
Docker
VS Code Dev Container
Rysen SDK environment
ROS2 Humble
Python SDK import
ROS2 colcon build
OpenCV
MediaPipe
NumPy
SciPy
pytest module
```

Rysen Python SDK 已验证：

```python
from rysen_apexhand_sdk import Rysen
bot = Rysen()
```

当前没有必要重新配置环境。

除非明确发生 dependency failure，否则不要修改 Docker 环境。

---

## M1 — Perception Baseline

状态：

```text
COMPLETED
```

已经打通：

```text
Physical Webcam
→ Windows
→ usbipd
→ WSL
→ /dev/video0
→ Docker
→ OpenCV
→ MediaPipe
```

当前摄像头：

```text
/dev/video0
```

已验证模式：

```text
MJPG
1280 × 720
30 FPS
```

MediaPipe 可以实时检测：

```text
21 hand landmarks
handedness
hand world landmarks
```

已有：

```text
perception/mediapipe_hand.py
visualization/hand_draw.py
scripts/run_hand_landmarks.py
```

可以生成带骨架的 annotated video。

MediaPipe inference 实测大约：

```text
~10–15 ms
```

具体数据以后应通过 benchmark 模块正式统计 p50/p95，而不是依赖当前日志。

---

## M2.1 — Palm Local Coordinate Frame

状态：

```text
IMPLEMENTED
```

目标：

```text
MediaPipe world landmarks
→ translation normalization
→ rotation normalization
→ hand-size normalization
→ palm-local landmarks
```

当前约定：

```text
origin = wrist landmark 0

+x:
pinky MCP → index MCP

+y:
wrist → middle MCP
并通过 Gram-Schmidt 与 x 正交

+z:
x × y
```

核心关系：

```text
p_local = R^T (p_world - origin)
```

NumPy row-vector implementation 可写作：

```python
centered @ rotation
```

之后除以 characteristic palm scale。

已有：

```text
human_hand/palm_frame.py
tests/test_palm_frame.py
```

至少应验证：

```text
R^T R = I
wrist becomes origin
translation invariance
```

后续必须补：

```text
rotation invariance test
scale invariance test
degenerate geometry test
```

---

## M2.2 — Human Index Joint Angles

状态：

```text
BASELINE IMPLEMENTED
NEEDS FINAL VALIDATION
```

当前优先只处理食指：

```text
5 MCP
6 PIP
7 DIP
8 TIP
```

目标输出：

```text
MCP abduction/adduction
MCP flexion/extension
PIP flexion
```

对应 Apex：

```text
index_j0
index_j1
index_j2
```

已有：

```text
human_hand/joint_angles.py
tests/test_joint_angles.py
```

当前真人实验已经观察到：

```text
PIP flexion 随真实屈指明显增大
MCP flexion 随真实屈曲明显变化
```

曾发现两个需要检查的问题：

```text
1. MCP flexion sign
2. MCP abduction zero/reference definition
```

不要假设修复已经提交。

必须检查当前代码。

正确方向：

MCP flexion：

```text
真实屈曲增加
→ calculated flexion 应增加为正值
```

MCP abduction 不应该简单测：

```text
proximal phalanx vs palm global +y
```

因为自然食指本身就与 +y 存在夹角。

更合理的 baseline 是：

```text
reference:
wrist → finger MCP 在 palm plane 的投影

current:
MCP → PIP 在 palm plane 的投影

signed angle:
atan2(cross_z, dot)
```

但必须通过真人实验验证。

---

# 8. Immediate Next Milestone

# M2.3 — Index Finger Calibration + Mapping

这是下一阶段的直接工作。

暂时禁止扩展到五指。

必须先把：

```text
Human index
→ Apex index
```

完整做通。

---

## M2.3.1 Human Index Calibration

实现 calibration 层。

禁止直接假设：

```text
human neutral angle = 0
```

真人伸直时可能：

```text
MCP abd ≠ 0
MCP flex ≠ 0
PIP flex ≠ 0
```

需要至少支持：

```text
neutral/open calibration
maximum MCP flexion
maximum PIP flexion
minimum/maximum abduction
```

建议数据结构：

```python
@dataclass
class JointRange:
    minimum: float
    maximum: float
```

以及：

```python
@dataclass
class IndexCalibration:
    human_abduction: JointRange
    human_mcp_flexion: JointRange
    human_pip_flexion: JointRange
```

但可以在不破坏架构的情况下提出更合理设计。

Calibration 必须：

```text
与算法解耦
可保存
可加载
可重复使用
```

后续建议保存：

```text
YAML / JSON
```

不要永远硬编码在 Python 文件。

---

## M2.3.2 Human → Apex Range Mapping

baseline 使用线性 range mapping：

```text
ratio =
(theta_h - theta_h_min)
/
(theta_h_max - theta_h_min)
```

裁剪：

```text
ratio ∈ [0, 1]
```

映射：

```text
q_robot =
q_min
+
ratio * (q_max - q_min)
```

必须支持：

```text
sign inversion
offset
clamping
invalid calibration rejection
```

所有内部值使用 radians。

---

## M2.3.3 Apex Index Target Type

不要直接在 pipeline 中传裸 tuple：

```python
(j0, j1, j2)
```

建议明确结构：

```python
@dataclass
class ApexIndexTarget:
    j0_abduction: float
    j1_mcp_flexion: float
    j2_pip_flexion: float
```

后续可以进一步统一为：

```text
ApexJointTarget
```

但当前不要过度抽象。

---

## M2.3.4 Tests

必须至少测试：

```text
source min → target min
source max → target max
source midpoint → target midpoint
below source min → clipped target min
above source max → clipped target max
invalid source range → exception
```

真人实验必须确认：

```text
Human left/right index swing
→ Apex j0 单调变化

Human MCP bend
→ Apex j1 单调增加

Human PIP bend
→ Apex j2 单调增加
```

---

# 9. M2.4 — Apex URDF Kinematics

M2.3 完成后进入。

目标：

```text
Apex index q
→ URDF
→ Forward Kinematics
→ fingertip pose
```

第一阶段只做：

```text
right index finger
```

或者根据当前比赛实际硬件选择 left/right。

必须从官方 URDF 读取：

```text
joint origin
joint axis
joint limits
mimic
link hierarchy
tip frame
```

不要自己重新构造一套假机器人。

---

## M2.4.1 FK Baseline

至少实现：

```text
p_tip = FK(q)
```

验证：

```text
q = neutral
q = j0 only
q = j1 only
q = j2 only
```

观察 fingertip trajectory 是否符合真实机械结构。

必须明确：

```text
URDF origin RPY
joint axis
parent/child transform
```

不能把 URDF axis 当成 world axis。

---

## M2.4.2 DIP Coupling

Apex index：

```text
j3 mimic j2 1:1
```

因此 FK 必须包含：

```text
q_j3 = q_j2
```

但 actuator command 不包含 j3。

明确区分：

```text
active joint space
mechanical joint space
```

---

# 10. M2.5 — Virtual Index Validation

目标：

```text
Camera
→ human index angles
→ calibrated mapping
→ Apex q
→ FK / virtual model
```

必须验证：

```text
方向
幅度
joint limit
continuity
monotonicity
```

只有食指链路完全可信后，才允许扩展五指。

---

# 11. M3 — Four Non-Thumb Fingers

将已经验证的 index 算法泛化到：

```text
index
middle
ring
pinky
```

MediaPipe indices：

```text
index:
5 6 7 8

middle:
9 10 11 12

ring:
13 14 15 16

pinky:
17 18 19 20
```

禁止复制四份几乎一样的函数。

应该抽象：

```python
compute_finger_angles(
    local_points,
    indices,
    reference_definition,
)
```

但每根手指 calibration 必须允许独立。

输出：

```text
4 fingers × 3 active DoF
= 12 DoF
```

---

# 12. M4 — Thumb

拇指必须独立设计。

禁止直接套用普通手指公式。

因为 Apex thumb 有：

```text
4 active DoF
```

并具有：

```text
CMC abduction
CMC rotation
CMC flexion
MCP flexion
```

MediaPipe thumb：

```text
1 CMC
2 MCP
3 IP
4 TIP
```

拇指需要重点处理：

```text
opposition
rotation
out-of-plane orientation
```

可能需要：

```text
vector basis
relative rotation
plane normal
signed angles
```

在数学定义明确前禁止凭经验堆 atan2。

---

# 13. M5 — Full Human Hand State

最终建立统一结构，例如：

```python
@dataclass
class HumanHandState:
    ...
```

包含：

```text
handedness
timestamp
confidence
joint angles
optional landmarks
```

不要让下游 retargeting 继续依赖 MediaPipe 原始对象。

必须建立：

```text
Perception API boundary
```

即：

```text
MediaPipe-specific representation
↓
HumanHandState
↓
retargeting
```

这样以后可以替换：

```text
MediaPipe
MANO
custom network
RGB-D
```

而不重写 retargeting。

---

# 14. M6 — Full Apex Joint Target

建立统一：

```text
16 active DoF
```

目标结构。

必须明确 canonical joint order。

禁止在多个文件里各自定义顺序。

推荐唯一 source of truth，例如：

```text
robot/apex_joint_names.py
```

或者：

```text
robot/model.py
```

所有：

```text
array index
ROS command
SDK command
logging
calibration
```

必须使用同一 ordering。

---

# 15. M7 — Filtering

只有 raw geometry 与 mapping 验证之后才进入。

优先比较：

```text
One-Euro Filter
EMA
Kalman Filter
```

第一 baseline 优先：

```text
One-Euro
```

因为适合：

```text
低速抑制 jitter
快速运动降低 lag
```

需要实验比较：

```text
noise reduction
phase delay
latency
overshoot
```

不要只看视频“感觉更稳”。

Filtering 层应独立于：

```text
MediaPipe
retargeting
robot SDK
```

---

# 16. M8 — Real-Time Architecture

当前 sequential prototype 最终要升级为：

```text
Capture Thread
↓ latest frame

Inference / Retarget Thread
↓ latest command

Control Thread
↓ robot
```

原则：

```text
latest-data semantics
```

禁止积累旧帧 FIFO。

如果视觉处理落后：

```text
drop stale frames
```

而不是：

```text
处理几秒前的 frame
```

需要 timestamp：

```text
capture timestamp
inference start/end
retarget end
command timestamp
device response timestamp
```

---

# 17. M9 — Safety Layer

真实硬件前必须实现。

至少：

```text
joint position clamp
joint velocity limit
joint acceleration limit
invalid pose rejection
confidence threshold
stale command detection
lost-hand behavior
emergency stop
```

当：

```text
MediaPipe lost hand
```

禁止机器人继续无限使用最后一个运动趋势。

应设计明确策略，例如：

```text
hold current position
```

或者：

```text
safe timeout → neutral
```

策略必须可配置并经过实机验证。

---

# 18. M10 — Robot Abstraction

在真实硬件之前建立统一接口。

例如：

```python
class ApexInterface(Protocol):
    def command_position(...):
        ...

    def get_joint_state(...):
        ...

    def close(...):
        ...
```

至少实现：

```text
MockApexHand
RealApexHand
```

所有视觉和 retargeting 代码禁止直接 import vendor SDK。

只有：

```text
robot/rysen_backend.py
```

这一层可以直接依赖：

```python
rysen_apexhand_sdk
```

这样才能在没有硬件时测试整个 pipeline。

---

# 19. M11 — ROS2 / Rysen Integration

官方栈关系：

```text
Rysen Retargeting
→ ROS2 command topic
→ Rysen Explorer backend
→ Rysen SDK
→ Apex Hand
```

本项目可以选择：

```text
A. 直接通过 Rysen Python SDK
```

或：

```text
B. 发布 ROS2 joint command
→ Rysen backend
```

不要过早决定。

先做接口 benchmark，比较：

```text
latency
reliability
deployment complexity
competition requirements
```

当前官方 Rysen Retargeting 使用 MANUS glove。

本项目视觉系统不能简单把官方 retargeting 当作自主算法。

---

# 20. M12 — Real Hardware Bring-up

真实 Apex Hand 到手后严格按照以下顺序：

```text
network
→ ping
→ SDK connect
→ read-only status
→ enable
→ one finger
→ one joint
→ low velocity
→ low range
→ multi-joint
→ vision command
```

禁止第一次连接就运行完整视觉 teleoperation。

先限制：

```text
速度
加速度
关节范围
```

使用远低于硬件极限的值。

---

# 21. M13 — Calibration Procedure

最终系统应有明确 calibration workflow：

```text
camera configuration
human neutral pose
human ROM
handedness
robot zero
retargeting scale
sign verification
```

Calibration 必须持久化。

建议：

```text
configs/calibration/*.yaml
```

不要每次运行重新硬编码。

---

# 22. M14 — Metrics and Benchmarking

最终必须能够自动记录：

```text
FPS
inference latency
retarget latency
control latency
end-to-end latency
jitter
dropped frames
lost detection rate
```

统计至少：

```text
mean
median
p95
max
```

不要只汇报：

```text
Average FPS
```

---

## Accuracy Metrics

可以计算时使用：

```text
human joint angle repeatability
robot target repeatability
trajectory correlation
time lag
overshoot
steady-state jitter
```

若有 ground truth，增加：

```text
angle MAE
fingertip position error
orientation error
```

没有 ground truth 时禁止把自定义代理指标称为“真实姿态误差”。

---

# 23. M15 — Occlusion and Robustness

系统必须测试：

```text
self-occlusion
finger overlap
fast motion
hand rotation
partial out-of-frame
lighting changes
background changes
```

当 landmarks 不可信时，不允许继续发送高速度 command。

需要：

```text
confidence gating
temporal consistency check
outlier rejection
```

---

# 24. M16 — Optional Advanced Retargeting

只有 baseline 完成后才考虑：

```text
task-space optimization
fingertip matching
direction matching
MANO
temporal models
learned retargeting
```

可能目标：

```text
q* = argmin
λ1 L_tip
+ λ2 L_direction
+ λ3 L_angle
+ λ4 L_smooth
```

约束：

```text
joint limits
velocity limits
acceleration limits
```

不要为了“用了 AI”而加入神经网络。

比赛 baseline 应优先保证：

```text
deterministic
low latency
interpretable
calibratable
stable
```

---

# 25. Testing Requirements

所有核心数学模块必须有 unit tests。

最低要求：

```text
palm_frame
joint_angles
range_mapping
joint_limits
mimic joints
FK
filtering
safety clamp
timestamp logic
```

必须使用 synthetic geometry 测试。

例如：

```text
straight finger → flexion ≈ 0
45° artificial bend → flexion ≈ 45°
translation → local coordinates invariant
rotation → local representation invariant
```

不能只拿真人摄像头“看起来没问题”当作测试。

---

# 26. Logging

后续建立统一 logging。

禁止长期保留：

```python
print(...)
```

散落在核心模块。

脚本/debug 可以 print。

核心 library 使用：

```python
logging
```

实时 loop 禁止每帧输出大量日志。

---

# 27. Performance Rules

实时 loop 中禁止：

```text
反复加载模型
反复读取 URDF
每帧创建昂贵 solver
每帧打开文件
每帧初始化 filter
```

必须：

```text
initialize once
reuse every frame
```

大数组操作优先 NumPy vectorization。

但在 profiling 之前不要做无意义 micro-optimization。

---

# 28. Git Workflow

每个 milestone 结束后形成一个独立、可运行状态。

建议提交粒度：

```text
feat(perception): add MediaPipe hand detector
feat(geometry): add palm local frame
feat(geometry): add finger angle solver
feat(retarget): add index calibration mapping
feat(robot): add Apex URDF model
feat(filter): add One-Euro joint filtering
...
```

不要将：

```text
Docker
MediaPipe
FK
ROS
Filtering
```

全部堆在一个 commit。

---

# 29. Required Behavior

每次接到新任务时必须：

1. 读取本 CLAUDE.md。
2. 检查当前 git diff。
3. 检查相关现有模块。
4. 检查相关 tests。
5. 确认当前 milestone。
6. 只修改完成当前 milestone 所需的最少文件。
7. 写实现。
8. 写/更新 tests。
9. 运行 tests。
10. 报告结果。
11. 不自动进入下一个 milestone，除非用户明确要求。

禁止：

```text
顺手重构整个项目
升级所有依赖
修改 vendor repositories
加入未经要求的新框架
加入不必要的深度学习模型
把实验参数硬编码进算法
跳过测试
```

---

# 30. Response Format

完成每个开发任务后，必须报告：

```text
Milestone:
Status:

Modified files:

Implementation:
- ...

Tests:
- command
- result

Engineering decisions:
- ...

Known limitations:
- ...

Next recommended step:
- ...
```

如果测试失败：

不要隐藏。

报告：

```text
failed test
error
suspected cause
what remains unresolved
```

---

# 31. Current Immediate Task

当前不要实现五指。

先完成：

```text
M2.3 Index Finger Calibration + Apex Mapping
```

执行顺序：

```text
1. Inspect existing M2.1/M2.2 code.
2. Confirm MCP flexion sign from current implementation.
3. Confirm MCP abduction reference implementation.
4. Run existing tests.
5. Add missing M2.2 regression tests if needed.
6. Design calibration dataclasses.
7. Implement reusable range mapping.
8. Add Apex index target representation.
9. Add unit tests.
10. Integrate into live camera script only after unit tests pass.
11. Print/log Human Index and Apex Index values at low frequency.
12. Stop.
```

M2.3 完成并经过真人验证前：

```text
DO NOT IMPLEMENT OTHER FINGERS.
```

---

# 32. Final Target Architecture

最终软件架构应收敛为：

```text
                     Camera
                       │
                       ▼
                Frame Source
                       │
                       ▼
                 Perception
                       │
                 Human landmarks
                       │
                       ▼
            Human Hand Representation
                       │
                 HumanHandState
                       │
                       ▼
                  Retargeting
                       │
                 ApexJointTarget
                       │
                       ▼
                    Filter
                       │
                       ▼
                    Safety
                       │
                       ▼
                 Robot Interface
                  /           \
                 /             \
         Mock Apex           Real Apex
                               │
                         Rysen SDK / ROS2
                               │
                               ▼
                           Apex Hand
```

依赖方向必须保持：

```text
perception
    ↓
human_hand
    ↓
retargeting
    ↓
filtering / safety
    ↓
robot
```

下层禁止反向 import 高层。

特别是：

```text
human_hand
```

不能知道：

```text
Rysen SDK
ROS2
Apex hardware networking
```

而：

```text
perception
```

不能知道：

```text
Apex joint limits
```

保持模块职责严格隔离。

---

# 33. Definition of Done

整个比赛项目只有满足以下条件才视为工程完成：

```text
Camera realtime capture              PASS
21-landmark perception               PASS
Palm local coordinates               PASS
Human joint reconstruction           PASS
Human calibration                    PASS
16-DoF retargeting                   PASS
URDF FK validation                   PASS
Filtering                            PASS
Safety constraints                   PASS
Mock robot integration               PASS
Real Apex SDK integration            PASS
Real-time multi-thread pipeline      PASS
Latency benchmark                    PASS
Robustness benchmark                 PASS
Calibration persistence              PASS
Repeatable startup/deployment        PASS
Documentation                        PASS
```

最终必须能够通过一个明确入口启动系统，例如：

```bash
python3 -m apex_retargeting.run --config configs/apex_right.yaml
```

具体 module name 可在后续架构整理时确定。

禁止最终比赛版本依赖：

```text
手工改源码参数
手工复制文件
随机执行几个测试脚本
临时 pip install
```

最终 deployment 必须：

```text
reproducible
documented
config-driven
testable
```