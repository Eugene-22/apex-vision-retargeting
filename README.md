# Apex Vision Retargeting

将相机或视频中的人手姿态映射为 Rysen ApexHand 关节角。实现包含 `perception`、`retargeting` 和右手真机控制层 `robot`，以及相机、视频和关键点回放入口。输出为 SDK 关节顺序的 **21 维弧度角**；URDF 中的五组 mimic 联动通过 **16 个独立变量**求解。

## 安装与运行

在项目根目录执行（Python 3.10+；厂商 SDK 的二进制目前要求 Python 3.10）：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[vision,test]"
python -m perception.download_model
python -m retargeting --camera 0 --hand right --show
```

按 `q` 或 Esc 退出预览。默认假设输入为未镜像相机画面；若输入已镜像，加 `--input-mirrored`。使用的是 MediaPipe Tasks API 和独立下载的模型，不依赖已经移除的 `mp.solutions` 接口。模型下载是显式操作，检测器初始化不会联网。

```bash
# 右手；结果保存到 JSONL
python -m retargeting --camera 0 --hand right --output joints.jsonl

# 视频输入，录制可复现的 metric world landmarks
python -m retargeting --video hand.mp4 --record landmarks.jsonl --output joints.jsonl

# 回放无需 MediaPipe、OpenCV、模型或相机
python -m retargeting --replay landmarks.jsonl --optimizer adaptive --output replay_joints.jsonl

# 对比普通关键向量模式
python -m retargeting --replay landmarks.jsonl --optimizer vector --output vector_joints.jsonl
```

回放按记录的时间戳计算滤波与速度约束，以最快速度处理，不按真实时间等待。每条输出包含时间戳、左右手、`tracked`、`valid`、`joint_ids`、`joint_names`、`qpos` 和求解状态。未检测到目标手、输入退化或求解失败时，`qpos` 为 `null`，不重复发布旧角度。

`python -m retargeting` 只生成关节角。右手真机入口是 `python -m robot`，见 [robot 使用说明](robot/README.md)。该入口默认空跑，显式指定 `--execute` 和实际设备 IP 后才连接硬件；通过设备反馈确认右手，并从实测关节角开始限速跟随。

## 感知数据约定

- `HandDetector.detect(bgr_frame, timestamp)` 输入 `uint8 (H,W,3)` BGR 图像与严格递增的秒时间戳，返回 `HandObservation` 或 `None`。
- `HandObservation.keypoints` 是 MediaPipe 标准顺序的 `(21,3)` **米制 world landmarks**。它们是单目模型估计的三维坐标，不是深度相机测量值；其全局平移不用于机器人腕部控制。
- `image_landmarks` 仅用于原始输入画面上的叠加显示，不能作为米制优化输入。
- `score` 是左右手分类置信度，不代表全部关键点的精度。
- 检测器内部处理镜像、BGR/RGB 转换和毫秒时间戳，并只选择所请求的左右手。相机资源、检测器及输出文件均使用上下文管理器关闭。
- `ObservationWriter` / `ReplaySource` 以带版本号的 JSONL 保存/读取关键点及丢失帧；读取时验证形状、有限数值、时间顺序，不使用 pickle。

## 重定向方法

手掌坐标转换、关键向量目标、自适应捏合混合、解析梯度和低通滤波思路。这里按 ApexHand 模型独立实现运动学与目标函数，使用 NumPy/SciPy SLSQP，避免运行时依赖 Pinocchio/NLopt。

1. 根据 wrist/index MCP/middle MCP 建立手掌坐标系，移除输入的整体平移和旋转，应用左右手方向变换；共线或坍缩的手掌被拒绝。
2. 默认根据腕到中指 MCP 的长度匹配机器人尺度，将普通手指 MCP 锚定到机器人基座，并按机器人骨长逐段重建目标。拇指 CMC 保留可动目标。配置可关闭骨长匹配，或调整每段比例与 XYZ 旋转。
3. `vector` 模式匹配 wrist→各手指中间关节/指尖的 15 个向量，使用 Huber 距离损失和相邻帧关节正则项。
4. `adaptive` 模式根据原始人手拇指到其他指尖的距离渐变权重，在全手目标与指尖位置/末节方向之间混合；当前最强捏合对还增加接触距离惩罚。捏合目标使用统一缩放，避免各指独立骨长缩放破坏接触。它是几何目标，没有接触动力学或碰撞模型。
5. 从实际 URDF 解析正向运动学、局部点偏移、关节限位和 mimic 关系。解析雅可比通过链式法则映射到 16 个独立变量，优化后展开为 SDK 的 21 个关节角。
6. 用上帧输出热启动；输出经过与时间间隔相关的低通滤波和速度约束。超过 `tracking_timeout` 的丢失间隔清除滤波及优化状态。求解未收敛但达到迭代上限且目标改善时，可以接受有界结果；`success` 与 `accepted` 分开报告，调用者可以选择只接受收敛帧。

左手 URDF 部分 tip/pad 的名称与实际位置不一致。因此两手都使用末节连杆局部偏移定义指尖：拇指 32 mm，其他四指 28 mm，可在配置中调整。

## Python 接口

```python
from perception import HandDetector, VideoSource
from retargeting import Retargeter

retargeter = Retargeter.from_yaml("config/apex_hand.yaml", hand_side="right")
with VideoSource(0) as camera, HandDetector("models/hand_landmarker.task") as detector:
    for timestamp, frame in camera:
        observation = detector.detect(frame, timestamp)
        qpos = retargeter.process(observation, timestamp)
        if qpos is not None:
            print(qpos)  # shape (21,), radians; no hardware command is sent
```

离线已有关键点时，可调用 `retargeter.retarget(points)` 或 `retarget_verbose(points)`。原始输入是米制 world landmarks；已经处于 URDF 手掌坐标且完成标定的目标可显式传 `input_frame="robot"`，跳过坐标转换、尺寸归一化和骨长重建。`apply_filter=False` 用于优化诊断，会同时跳过输出滤波和逐帧速度约束。实时控制建议使用会为丢失/无效帧返回 `None` 的 `process` 接口，并持续传入时间戳。

`ApexHandModel.keypoints(qpos, with_jacobian=True)` 返回 21 个机器人对应点以及 `(21,3,21)` 的完整关节雅可比。输入完整 21 维角度时该低层函数不会自动应用 mimic，方便数值微分；实际优化通过 `expand()` 强制满足 mimic。

## 配置与关节顺序

主配置为 `config/apex_hand.yaml`，URDF 路径相对该文件解析。两手共用参数，`{hand_side}` 选择 URDF，`rotation_degrees` 支持左右手分别标定。默认值是几何初始化，仍需根据实际镜头、人手比例和机器人指尖结构调整。

| SDK ID | URDF 关节 | 含义 |
| --- | --- | --- |
| 0–4 | f0_joint0–4 | 拇指 CMC 外展、旋转、屈曲；MCP、IP 屈曲 |
| 5–8 | f1_joint0–3 | 食指 MCP 外展、MCP/PIP/DIP 屈曲 |
| 9–12 | f2_joint0–3 | 中指，顺序同上 |
| 13–16 | f3_joint0–3 | 无名指，顺序同上 |
| 17–20 | f4_joint0–3 | 小指，顺序同上 |

## 验证

```bash
python -m pytest -q
```

测试覆盖左右手 FK 几何、全部关节解析雅可比的中心差分、两种损失的梯度、可达姿态恢复、指尖接触改善、限位/限速、联动、非法输入和失败回退、手掌刚体变换不变性、镜像处理、录制回放及命令行端到端输出。摄像头画面质量、真实人手标定和实体机器人运动需要在对应设备上验证。

真实模型集成测试需要额外指定官方图片和模型路径，否则默认跳过这两项：

```bash
APEX_TEST_HAND_IMAGE=/path/to/right_hands.jpg \
APEX_TEST_MODEL=/path/to/hand_landmarker.task \
python -m pytest -q
```

图片使用 [MediaPipe 官方 right_hands.jpg 测试素材](https://storage.googleapis.com/mediapipe-assets/right_hands.jpg)，测试不会自行下载。2026-09-06 本机验证结果：36 项数值/接口测试与 2 项真实模型集成测试全部通过；由官方图片与空帧组成的三帧视频成功完成感知、重定向、录制及回放，回放角度与视频处理角度一致，空帧输出 `null`。测试环境为 Python 3.10.12、NumPy 2.2.6、SciPy 1.15.3、MediaPipe 1.0.1、OpenCV 5.0.0.93。SLSQP 在部分试探步会提示裁剪越界中间值；最终输出的限位和联动均通过断言。

30 帧合成连续姿态的优化耗时中位数：右手约 8.6 ms、左手约 9.0 ms；该数字只包含优化，不包含相机采集、神经网络推理与预览，不是整条链路的实时帧率承诺。

当前 Docker 配置仍可用于离线回放：在容器中安装依赖后执行以上命令。相机预览还需要按实际设备配置视频设备映射及显示权限，现有 compose 未配置这些内容。

MediaPipe API 依据 [官方 Hand Landmarker Python 文档](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python)。
