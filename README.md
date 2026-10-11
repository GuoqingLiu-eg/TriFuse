# TriFuse

面向 Apple 统一内存设备的 CPU–GPU 推理运行时。CPU 管理请求、计算提交和状态生命周期，GPU 通过原生 MLX 路径执行模型计算。

**本仓库只保留运行框架所需源码、构建配置和模型导出工具。** 不包含 benchmark、对照框架、旧 App 探针、训练代码、实验报告、模型权重或编译产物。

## 运行时组成

| 路径 | 内容 |
|---|---|
| `include/trifuse/apple/` | Objective-C / Swift 可调用的会话接口 |
| `src/apple/` | C++ 会话桥接、权重数据加载和可选压缩执行器 |
| `include/trifuse/kv/` | KV 所有权、阈值维护、候选发布与压缩 attention 辅助 |
| `overlays/executorch/` | MLX 原生模型适配、提交策略和状态管理新增源码 |
| `patches/` | 对固定版本 ExecuTorch 和 MLX 的必要修改 |
| `tools/` | 依赖恢复、Apple 库构建、模型导出 |

默认主计算使用 GPU；可选的冷 KV 维护保留完整的压缩、发布和消费路径。压缩能力目前限定于已经实现的 OLMoE KV 几何，不能对所有模型直接开启。Core ML 仅用于该可选组件，不代表一定在 ANE 执行。

## 构建

需要 Apple Silicon Mac、Xcode/iPhoneOS SDK 和 Python 3.12。当前 iOS 目标为 arm64、iOS 26.2 或以上；Jetson/CUDA 后端尚未实现。

```sh
git clone https://github.com/GuoqingLiu-eg/TriFuse.git
cd TriFuse
python3.12 -m venv .venv
.venv/bin/python -m pip install --no-cache-dir -r requirements-build.txt
.venv/bin/python tools/bootstrap.py --submodules
.venv/bin/python tools/build_apple.py --platform ios --python .venv/bin/python
```

`bootstrap.py` 根据 `upstream-lock.json` 以浅克隆恢复上游，再应用两份补丁和本地新增源码。它不会下载模型或操作设备；重复运行不会覆盖不同的本地源码。上游本体不重复入仓。

构建失败后可用同一命令重试：脚本会替换自己标记的未完成打包目录，并复用已编译的依赖。成功产物不会被自动覆盖。

`build_apple.py --plan` 可查看构建步骤；`--platform macos` 选择 macOS。构建产物包括 `libTriFuse.a`、需要链接的依赖库清单、公开头文件和 MLX Metal 资源。将链接清单对应的库加入 App，并将资源 bundle 复制进 App；iOS 签名由使用方的 Xcode 工程配置。

已从 GitHub 全新克隆，在独立 Python 3.12.14 环境中安装依赖、恢复固定版本源码，并完成 iOS arm64 与 macOS arm64 的完整编译和链接，未复用原工程的预编译库。验证环境为 macOS 26.6、Xcode SDK 26.5，部署目标为 26.2；iOS 公开模块的 Swift 导入检查通过。

Mac 上还完成了合成小型 OLMoE 的模型导出与实际 GPU 运行，覆盖 prefill、decode、默认异步生成及 KV 长度/状态推进检查。这是可运行性检查，不代表真实模型的质量或性能结果；本次未执行手机真机推理，可选冷 KV 压缩仅验证了编译和链接。

## 接入会话

公开接口为 `TFETResidentSession`，见 [`TFETResidentSession.h`](include/trifuse/apple/TFETResidentSession.h)。可通过模块头或 Swift bridging header 导入。

运行顺序：

1. 加载导出模型的 manifest、prefill/decode `.pte` 和公共 `.ptd` 权重，按其配置创建会话。
2. 设置请求的 `TFETNativeExecutionPermit`，调用 `beginRequest` 确定本次 KV 容量。
3. 对已支持的 Granite/OLMoE，调用 `beginPythonPolicy`，通过 `nextGreedyPipeline` 获取 token，再调用 `finishGreedyPipeline` 完成排空并取得新的 generation。
4. 正常达到生成长度时恢复最后一次 yield；EOS/提前结束遵循接口中的提前关闭规则。出现执行失败时按接口返回的失效/退役状态处理，不继续复用未知完成状态。

`beginPythonPolicy` 是既有调度算法的 **C++ 实现**，运行时不依赖 Python。其他原生模型接口仍保留在后端；不要将 MoE 调度入口直接用于不支持的架构。

所有 MLX 会话操作须在同一专用线程串行执行。`cancel()` 可从其他线程撤销许可；iOS App 离开前台时，应撤销许可并让现有取消路径排空已提交工作。框架库不包含 UIKit 前台监测或界面代码，由宿主 App 接入。

## 模型准备

提供 OLMoE 和 Granite MoE 的原生导出工具，保留已有权重的量化表示。当前 OLMoE 源版本是：

```text
mlx-community/OLMoE-1B-7B-0125-Instruct-4bit
e23844197887b031e7ddddbb0b8959c5a6853a7b
```

先自行取得该版本权重及配置，使用 `tools/export/prepare_model.py` 对 OLMoE 专家权重做原字节堆叠，然后运行 `export_native_olmoe_target.py`。各工具均提供 `--help`；`prepare_model.py --self-test` 只验证合成字节转换，不加载模型。

可选压缩器由 `tools/export/export_affine_codec.py` 生成，通过 `TRIFUSE_COLD_KV_MODEL` 指定模型路径；`TRIFUSE_COLD_KV_CPU_ONLY=1` 使用其 CPU 路径。未配置压缩模型时保持普通 KV 路径。权重和导出产物均不入仓。

## 来源

ExecuTorch、MLX 和模型适配所依据的 MLX-LM 许可证保留在 `LICENSES/`；现有源码版权声明保留。此源码发布没有为全部原创代码另行指定统一许可证。
