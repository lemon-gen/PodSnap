# PodSnap
Summarize, highlighting the main content, innovations, and conclusion statements

一个基于大语言模型（LLM）的播客智能摘要工具，支持自动音频转录、核心内容提取、多风格总结、文案生成，快速获取播客精华内容。

## ✨ 核心功能
- 🎙️ 支持本地播客音频导入
- 📝 自动语音转文字生成完整文稿
- 🔍 基于 LLM 智能提取播客核心观点、大纲、金句
- 📄 生成精简摘要、详细总结、速览版文案
- 📊 支持导出 TXT / Markdown / JSON 格式报告
- ⚡ 轻量易部署，支持本地离线运行

## 📋 环境配置
### 前置依赖
- Python 3.10.16
- 推荐使用虚拟环境（conda）

### 安装步骤
# 1. 创建并激活虚拟环境（可选但推荐）
# conda方式
```
conda create -n PodSnap python=3.10.16
conda activate PodSnap
```
# 2. 安装依赖包
```
pip install -r requirements.txt
```
# 3. 临时配置清华镜像源（if you need it）
```
export HF_ENDPOINT=https://hf-mirror.com
```

## Quickstart
Below, we provide simple examples to show how to use PodSnap in demo.

```
conda activate Podsnap
```
then,enter demon and run StoW_test1.py to test the ability of change mp4 into words.
```
cd demo
python3 StoW_test1.py
```
