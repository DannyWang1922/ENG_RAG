# ELC1012 RAG 系统

这是一个基于CrewAI框架的RAG（检索增强生成）系统，专门用于处理ELC1012课程材料。系统能够智能地检索和分析课程内容，为用户提供准确、详细的学术信息。

## 功能特点

- **智能文档处理**: 自动扫描ELC1012文件夹中的所有PDF文档，支持递归处理子文件夹
- **向量化存储**: 使用OpenAI的text-embedding-ada-002模型和Chroma向量数据库存储文档嵌入
- **多代理协作**: 使用CrewAI的研究代理和写作代理协作处理用户查询
- **智能检索**: 基于语义相似度检索相关课程内容
- **结构化回答**: 提供清晰、准确、结构化的回答
- **持久化存储**: 支持向量数据库的持久化，避免重复处理
- **文档更新检测**: 自动检测文档更新并重新构建向量数据库

## 系统架构

### 主要组件

1. **文档加载器**: 使用PyPDFLoader递归扫描ELC1012文件夹，加载所有PDF文档
2. **文本分割器**: 使用RecursiveCharacterTextSplitter将文档分割成适合向量化的块
3. **向量数据库**: 使用Chroma存储文档嵌入，支持持久化
4. **RAG工具**: 自定义RAG工具，用于检索相关信息
5. **代理系统**: 
   - 研究代理：负责检索和分析信息，使用gpt-4o-mini模型
   - 写作代理：负责组织和呈现信息，使用gpt-4o-mini模型

### 工作流程

1. 用户输入查询
2. 研究代理使用RAG工具检索相关课程内容
3. 研究代理分析检索到的信息
4. 写作代理基于分析结果生成结构化回答
5. 返回最终答案给用户

## 安装和设置

### 环境要求

- Python 3.8+
- OpenAI API密钥

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置OpenAI API密钥

有两种方式配置API密钥：

**方式1：环境变量**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

**方式2: main.py文件**
创建.main.py文件并添加：
```
OPENAI_API_KEY=your-api-key-here
```

### 确保文档结构

确保ELC1012文件夹包含所有课程材料的PDF文件，系统开发时的文档结构：

```
ELC1012/
├── ELC1012_1013.pdf
├── Assessment 2 Guide.pdf
├── Assessment 3 Guide.pdf
├── lesson1/
├── lesson2/
├── lesson3/
├── lesson4/
├── lesson5/
├── lesson6/
├── lesson7/
├── lesson8/
├── lesson10/
└── lesson11/
```

## 使用方法

### 运行系统

```bash
python main.py
```

### 使用流程

1. 启动程序后，系统会自动加载ELC1012文件夹中的所有PDF文档
2. 系统会创建向量数据库（首次运行可能需要一些时间）
3. 系统准备就绪后，你可以开始提问
4. 输入你的问题，系统会基于课程材料提供回答
5. 输入 'quit' 或 'exit' 退出程序

### 示例问题

- "什么是学术写作的引用格式？"
- "如何写一个有效的论文引言？"
- "Assessment 2的要求是什么？"
- "学术写作中如何避免抄袭？"
- "ELC1012课程的主要内容是什么？"

### 编程接口使用

```python
from main import ELC1012RAGSystem

# 初始化系统
rag_system = ELC1012RAGSystem()
rag_system.initialize_system()

# 处理查询
result = rag_system.process_query("什么是学术写作？")
print(result)
```

## 技术细节

### 文档处理

- 使用PyPDFLoader加载PDF文档
- 使用RecursiveCharacterTextSplitter进行文本分割
- 块大小：1000字符，重叠：200字符
- 支持递归扫描子文件夹

### 向量化

- 使用OpenAI的text-embedding-ada-002模型
- 使用Chroma作为向量数据库
- 持久化存储到./chroma_db目录
- 自动检测文档更新并重新构建数据库

### 模型配置

系统使用以下模型配置：

- **研究代理**: gpt-4o-mini
- **写作代理**: gpt-4o-mini
- **嵌入模型**: text-embedding-ada-002

### 代理配置

- **研究代理**: 专注于检索和分析课程内容，提供详细的分析报告
- **写作代理**: 专注于组织和呈现信息，生成结构化回答
- 使用顺序处理模式确保信息流的一致性

## 配置说明

系统配置在`config.py`文件中，主要配置项包括：

### 文档处理配置
```python
DOCUMENT_CONFIG = {
    "folder_path": "ELC1012",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "supported_formats": [".pdf"]
}
```

### 向量数据库配置
```python
VECTOR_DB_CONFIG = {
    "persist_directory": "./chroma_db",
    "collection_name": "elc1012_documents",
    "embedding_model": "text-embedding-ada-002",
}
```

### 代理配置
```python
AGENT_CONFIG = {
    "researcher": {
        "role": "ELC1012 Course Research Expert",
        "goal": "Deeply analyze ELC1012 course materials...",
        "llm_model": "gpt-4o-mini"
    },
    "writer": {
        "role": "Academic Writing Assistant", 
        "goal": "Organize retrieved information...",
        "llm_model": "gpt-4o-mini"
    }
}
```


## 项目结构

```
ENG_RAG/
├── main.py                 # 主程序文件
├── config.py               # 配置文件
├── requirements.txt        # 依赖包列表
├── README.md              # 项目说明
├── PROJECT_SUMMARY.md     # 项目总结
├── chroma_db/             # 向量数据库存储目录
└── ELC1012/              # 课程材料文件夹
    ├── lesson1/
    ├── lesson2/
    ├── ...
    ├── ELC1012_1013.pdf
    ├── Assessment 2 Guide.pdf
    └── Assessment 3 Guide.pdf
```
