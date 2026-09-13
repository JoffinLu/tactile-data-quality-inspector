# tactile-data-quality-inspector

对机器人触觉数据（RCT）做统计质量评估的 Python 项目。

- 包名：`tactile-qc`（源码位于 `src/tactile_qc/`）
- 版本：0.1.0
- 状态：Stage 1 — 项目骨架已初始化

## 项目结构

```text
tactile-data-quality-inspector/
├── README.md
├── pyproject.toml
├── requirements.txt
├── src/
│   └── tactile_qc/
│       ├── __init__.py
│       ├── io.py              # RCT 数据加载
│       ├── quality.py         # 质量指标计算
│       ├── anomaly.py         # 异常检测
│       ├── report.py          # 报告生成
│       └── cli.py             # 命令行入口
├── app.py                     # Streamlit Dashboard
├── notebooks/
│   └── 01_data_exploration.ipynb
├── tests/
│   └── test_quality.py
└── data/                      # 数据集放这里（gitignore）
```

## 快速开始

```bash
# 创建并激活虚拟环境
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

# 安装依赖与本包（可编辑模式）
pip install -r requirements.txt
pip install -e .
```

## 使用

```bash
tactile-qc --help             # CLI 入口
streamlit run app.py          # Dashboard
pytest                        # 运行测试
```
