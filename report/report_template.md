# Mini-BLIP2 图像描述生成复现实验报告

## 1. 论文信息

- 论文名称：BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models
- 论文地址：https://arxiv.org/abs/2301.12597

## 2. 任务说明

本实验复现的任务是图像描述生成（Image Captioning）。

- 输入：一张图像
- 输出：一句英文 caption

本次实现采用轻量化 Mini-BLIP2 结构，重点是跑通完整流程，而不是追求论文原始指标。

## 3. 数据集

- 数据集名称：Flickr8k
- 数据集地址：https://www.kaggle.com/datasets/adityajn105/flickr8k
- 实际使用数据量：前 200 张图像及其对应 caption

## 4. 模型结构

本次实现的整体结构如下：

```text
Image -> Frozen Vision Encoder -> Mini Q-Former -> Projection Layer -> Frozen Language Decoder -> Caption
```

### 4.1 Vision Encoder

- 使用模型：`openai/clip-vit-base-patch32`
- 处理方式：冻结参数，仅提取视觉 token 特征

### 4.2 Mini Q-Former

- query token 数量：16
- hidden size：256
- Transformer 层数：2
- attention heads：8
- 是否使用 cross-attention：是
- 作用：将视觉 token 压缩为适合语言模型使用的前缀表示

### 4.3 Projection Layer

- 作用：将 Mini Q-Former 输出映射到 OPT 词向量空间维度

### 4.4 Language Decoder

- 使用模型：`facebook/opt-125m`
- 处理方式：冻结参数，仅负责 caption 生成

## 5. 训练设置

- 训练数据量：前 200 张图像
- epoch：
- batch size：
- learning rate：
- optimizer：AdamW
- loss function：Cross Entropy Loss
- 冻结模块：Vision Encoder + Language Decoder
- 训练模块：Mini Q-Former + Projection Layer

## 6. 训练过程

在 `outputs/train_log.csv` 中记录每个 epoch 的训练损失与验证损失。

| Epoch | Train Loss | Val Loss |
|---|---:|---:|
| 1 |  |  |
| 2 |  |  |
| 3 |  |  |

可在此处补充训练日志截图或 loss 曲线图。

## 7. 生成结果展示

至少展示 3 个测试样本。

| 图片编号 | 真实 Caption | 模型生成 Caption |
|---|---|---|
| 1 |  |  |
| 2 |  |  |
| 3 |  |  |

如果方便，可以在报告中插入测试图片。

## 8. 总结

请说明：

- 是否成功跑通训练流程；
- 生成结果的大致效果；
- 遇到的主要问题；
- 后续可以如何改进。

## 9. AI 对话过程记录

- 录制工具：Entire / entir.io / 其他可分享工具
- 对话链接：
- 使用的 AI 模型：Codex / ChatGPT / Claude / Gemini
- 累计对话时长 / 会话数：

简要说明 AI 在哪些环节提供了帮助，以及你自己完成或修正了哪些内容：

```text
（在这里填写）
```

## 10. Git 提交记录

- 仓库地址：
- 总 commit 数：

粘贴 `git log --oneline` 输出：

```text
（在这里粘贴 git log --oneline）
```
