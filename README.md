# AI 文本嵌入与语义检索 MCP 服务

基于智谱 AI GLM 嵌入模型的文本嵌入与语义检索服务。支持作为 MCP 工具集使用，提供“文档索引 → 语义搜索”的完整演示闭环，并内置中文自然语言指令解析。

## ✨ 功能概览

- 文本嵌入
  - 单文本与批量嵌入（embedding-3 / embedding-2）
  - 文本相似度计算、候选集中相似文本搜索
- 知识库能力（新增）
  - 文档索引：支持 .md / .txt、本地目录递归与 http/https URL
  - 语义搜索：TopK 片段检索，返回来源与片段文本
  - 中文自然语言指令：`nl_command` 解析“索引/搜索”等口令
- 存储后端（免安装可用）
  - 自动检测 ChromaDB；不可用时回退到本地 JSONL 向量库（outputs/simple_kb）
- MCP 工具集
  - get_text_embeddings / get_batch_embeddings / calculate_text_similarity / find_similar_texts
  - get_supported_embedding_models / test_embedding_api / save_embeddings_to_file / load_embeddings_from_file
  - index_documents（新增）/ semantic_search（新增）/ nl_command（新增）

## 🚀 快速开始

### 环境要求
- Python 3.8+
- 智谱 AI API Key（环境变量：`ZHIPU_API_KEY`）
- Windows PowerShell 示例命令如下（其他平台同理调整）

### 安装依赖
可选其一：
- 使用 uv（推荐）：`uv sync`
- 或使用 pip（按需）：`pip install -U mcp[cli] fastapi uvicorn numpy requests`

> 说明：项目已内置本地向量库回退，无需安装 chromadb 即可演示。

### 配置环境变量（PowerShell）
```powershell
$env:ZHIPU_API_KEY="你的新Key"
# 限制可索引的本地路径（; 分隔多路径），建议开启
$env:ALLOW_INDEX_DIRS="D:\WorkProjects\AI\MCPServers\2AI-Embeddings\docs"
```

### 本地一行联调（无需 MCP 客户端）
```powershell
python -c "from main import index_documents, semantic_search; import json; \
print(json.dumps(index_documents(paths=['docs/sample/policies.md'], kb='kb_policies'), ensure_ascii=False, indent=2)); \
print(json.dumps(semantic_search(query='差旅报销怎么走', kb='kb_policies', top_k=3), ensure_ascii=False, indent=2))"
```

期望输出（示例，省略部分字段）：
```json
{
  "success": true,
  "kb": "kb_policies",
  "files_indexed": 1,
  "chunks_indexed": 2,
  "details": [ { "source": "docs\\sample\\policies.md", "chunks": 2, "status": "ok" } ]
}
{
  "success": true,
  "kb": "kb_policies",
  "query": "差旅报销怎么走",
  "top_k": 3,
  "results": [
    { "rank": 1, "source": "docs\\sample\\policies.md", "text": "…片段文本…" },
    { "rank": 2, "source": "docs\\sample\\policies.md", "text": "…片段文本…" }
  ]
}
```

### 启动 MCP 服务器
```bash
python main.py --mcp
```
在支持 MCP 的客户端（如蓝耘平台）选择此服务后，可在“工具”里调用：
- `nl_command`（自然语言路由）
  - `索引 docs\sample kb=kb_policies`
  - `搜索 "差旅报销怎么走" kb=kb_policies top=3`
- 或直接调用工具：
  - `index_documents(paths=["docs/sample/policies.md"], kb="kb_policies")`
  - `semantic_search(query="差旅报销怎么走", kb="kb_policies", top_k=3)`

## 🧰 MCP 工具列表（摘要）

- get_text_embeddings(input_text, model="embedding-3")
- get_batch_embeddings(texts, model="embedding-3")
- calculate_text_similarity(text1, text2, model="embedding-3")
- find_similar_texts(query_text, candidate_texts, model="embedding-3", top_k=5)
- get_supported_embedding_models()
- test_embedding_api(test_text=None)
- save_embeddings_to_file(texts, filename, model="embedding-3")
- load_embeddings_from_file(filename)
- index_documents(paths: List[str], kb="kb_default", chunk_size=500, overlap=50, model="embedding-3")  ← 新增
- semantic_search(query: str, kb="kb_default", top_k=5, model="embedding-3")  ← 新增
- nl_command(command: str)  ← 新增
  - 例：`索引 docs kb=kb_docs chunk_size=500 overlap=50`
  - 例：`搜索 "请假申请" kb=kb_docs top=5`

## 📦 存储后端说明

- 默认：内置本地向量库（JSONL + numpy 余弦检索），持久化于 `outputs/simple_kb/{kb}.jsonl`
  - 小规模（≤ 1 万 chunks）TopK 检索无压力，演示友好
- 可选：ChromaDB（自动检测）
  - 若本机已安装并可用，将自动使用 ChromaDB 持久化（无需修改调用层）
  - 若不可用，自动回退到本地 JSONL 后端

## 🔐 安全与约束

- 必需：`ZHIPU_API_KEY`（不要在终端/日志中明文回显）
- 路径白名单：`ALLOW_INDEX_DIRS` 限制可索引目录，防止越权读取
- 输入校验与错误处理：URL/路径存在性、网络异常重试、清晰的错误提示

## 📝 示例数据

已内置示例文档：`docs/sample/policies.md`  
可直接用于演示“索引/搜索”闭环。

## 🧱 项目结构

```
2AI-Embeddings/
├── main.py                         # 主入口（MCP 工具注册）
├── zhipu_embedding_client.py       # 智谱嵌入客户端
├── services/
│   ├── chunking.py                 # 文本分块（支持重叠）
│   ├── indexing.py                 # 文档收集与索引（复用嵌入、写入存储）
│   ├── searching.py                # 语义搜索（TopK 返回）
│   ├── vector_store.py             # 通用向量库接口（ChromaDB / JSONL 回退）
│   └── command_parser.py           # 中文自然语言指令解析（索引/搜索）
├── docs/
│   └── sample/policies.md          # 示例政策文档
└── outputs/
    └── simple_kb/                  # 本地向量库持久化（自动生成）
```

## 🔍 故障排除

- “无法读取本地路径”：设置 `ALLOW_INDEX_DIRS` 将目标目录加入白名单
- “chromadb 安装失败”：可忽略；系统已自动回退到本地 JSONL 后端
- “检索不准/片段不理想”：调参 `chunk_size` / `overlap`，或扩充语料
- “API 连接异常”：检查网络与 `ZHIPU_API_KEY`；可用 `test_embedding_api` 自检

## 📜 许可
MIT License

## 🙏 致谢
- 智谱 AI GLM 嵌入模型
- MCP 生态与开源社区