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
- epoch：3
- batch size：2
- learning rate：1e-4
- optimizer：AdamW
- loss function：Cross Entropy Loss
- 冻结模块：Vision Encoder + Language Decoder
- 训练模块：Mini Q-Former + Projection Layer
- 训练设备：NVIDIA GeForce RTX 4060 Laptop GPU

## 6. 训练过程

训练日志保存在 `outputs/train_log.csv` 中，结果如下：

| Epoch | Train Loss | Val Loss |
|---|---:|---:|
| 1 | 3.891864 | 3.008661 |
| 2 | 3.331450 | 3.006160 |
| 3 | 3.059652 | 2.902886 |

从结果可以看出，训练 loss 与验证 loss 都有下降，说明模型已经成功完成了一个可运行的训练流程。

## 7. 生成结果展示

以下展示 3 个测试样本：

| 图片编号 | 真实 Caption | 模型生成 Caption |
|---|---|---|
| 1000268201_693b08cb0e.jpg | A child in a pink dress is climbing up a set of stairs in an entry way . | A dog is playing with a toy . . . . . . . . . . . . . . . . . . . . . . . |
| 1001773457_577c3a7d70.jpg | A black dog and a spotted dog are fighting | A dog and a child are playing with a toy . . . . . . . . . . . . . . . . . . . . |
| 1002674143_1b742ab4b8.jpg | A little girl covered in paint sits in front of a painted rainbow with her hands in a bowl . | A man and a woman are standing on a beach . . . . . . . . . . . . . . . . . . . . |

虽然生成结果还比较粗糙，并且有明显的模式化重复，但模型已经能够根据图像输出英文文本，满足本次复现“流程跑通”的最低要求。

## 8. 总结

本次复现已经成功完成了以下工作：

- 成功读取 Flickr8k 前 200 张图像及对应 caption；
- 成功搭建 Mini-BLIP2 结构；
- 成功完成训练流程；
- 成功生成测试图像的 caption。

本次结果的主要问题是：

- 数据量非常小，只有 200 张图像，模型容易过拟合并产生模式化输出；
- 训练轮数较少，生成结果与真实语义仍有明显差距；
- 生成文本中存在重复句点和常见模板句式。

如果继续改进，可以考虑：

- 增加训练数据量，例如使用更多 Flickr8k 样本；
- 增加训练轮数并调节学习率；
- 优化 Q-Former 结构与 query token 数量；
- 改进生成策略，例如限制重复或调整 beam search 参数。

## 9. AI 对话过程记录

- 录制工具：Entire + Codex Desktop
- 对话链接：本次 Entire 仓库接入成功，但 Sessions / checkpoints 未被成功捕获，因此补充提交完整对话截图或导出文本作为过程证据
- 使用的 AI 模型：Codex
- 累计对话时长 / 会话数：本次开发过程在 Codex Desktop 中连续完成

说明：

本次开发过程中使用了 Codex 协助完成需求分析、数据读取、模型实现、训练脚本、生成脚本、环境排查和报告整理。  
在排查 Entire 记录功能时，已经成功完成了 GitHub App 安装、仓库连接、分支推送和 commit 识别，但在当前 Windows + Codex Desktop 环境下，Entire 的 Sessions / checkpoints 仍未被成功捕获。  
因此本次作业同时保留了：

- Git 小步提交记录；
- Entire 仓库接入页面截图；
- Entire Sessions 无 checkpoint 的截图；
- Codex 对话截图或导出文本；
- 训练日志与生成结果。

## 10. Git 提交记录

- 仓库地址：https://github.com/chentianbairizuomeng/blip2
- 总 commit 数：12

本次复现中与实现直接相关的关键提交如下：

```text
30dedce fix: align language decoder input dtype for training
7cd252b fix: rename package and enable CUDA training
89d1e17 docs: add usage guide and report template
94dd0bc feat: add training and caption generation scripts
d71ae10 feat: implement Mini Q-Former bridge model
e4b6c02 feat: load Flickr8k first 200 images and captions
c361f6c chore: enable Entire tracking for Codex
```

完整 `git log --oneline` 输出如下：

```text
30dedce fix: align language decoder input dtype for training
7cd252b fix: rename package and enable CUDA training
89d1e17 docs: add usage guide and report template
94dd0bc feat: add training and caption generation scripts
d71ae10 feat: implement Mini Q-Former bridge model
e4b6c02 feat: load Flickr8k first 200 images and captions
c361f6c chore: enable Entire tracking for Codex
b6011ba Translate README to Chinese and document data/ folder
edac08a Add empty data/ placeholder
674cc47 Add code/ placeholder so the directory appears in the repo
54a0028 Add anti-cheat requirements: AI chat log and granular git commits
d6cb42d Add Mini-BLIP2 reproduction brief
```
