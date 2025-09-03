"""
AI Text Embedding Generator - Main Entry Point

A comprehensive text embedding system supporting Zhipu GLM embedding models.
Provides both MCP server capabilities and direct text embedding functionality.
"""

import os
import sys
import time
import json
import uuid
import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from mcp.server.fastmcp import FastMCP
from zhipu_embedding_client import ZhipuEmbeddingClient
from network_diagnostic import NetworkDiagnostic
from services.indexing import collect_sources, index_to_chroma
from services.searching import search_kb
from services.command_parser import parse_command

# Create an MCP server
mcp = FastMCP("AI Text Embedding Generator")

# Create directories for storing files
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

# Load configuration
def load_config():
    """加载配置文件"""
    config_path = "config.json"
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

config = load_config()

# Initialize clients
api_key_str: str = (config.get("zhipu_api_key") or os.getenv("ZHIPU_API_KEY") or "")
embedding_base_url = config.get("text_embedding", {}).get("base_url", "https://open.bigmodel.cn")
embedding_client = ZhipuEmbeddingClient(api_key=api_key_str, base_url=embedding_base_url)

# Text Embedding Entry Point
class EmbeddingGenerator:
    """主要的文本嵌入入口类"""
    
    def __init__(self):
        self.embedding_client = embedding_client
        self.outputs_dir = OUTPUTS_DIR
    
    def get_embeddings(self, 
                      input_text: Union[str, List[str]],
                      model: str = "embedding-3") -> Dict[str, Any]:
        """
        主要的文本嵌入入口
        
        Args:
            input_text: 输入文本，可以是单个字符串或字符串列表
            model: 使用的嵌入模型 (embedding-3, embedding-2)
            
        Returns:
            嵌入结果
        """
        return self.embedding_client.get_embeddings(
            input_text=input_text,
            model=model
        )
    
    def get_single_embedding(self, text: str, model: str = "embedding-3") -> List[float]:
        """获取单个文本的嵌入向量"""
        return self.embedding_client.get_single_embedding(text, model)
    
    def get_batch_embeddings(self, texts: List[str], model: str = "embedding-3") -> List[List[float]]:
        """批量获取文本嵌入向量"""
        return self.embedding_client.get_batch_embeddings(texts, model)
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """计算两个嵌入向量的余弦相似度"""
        return self.embedding_client.calculate_similarity(embedding1, embedding2)
    
    def find_most_similar(self, query_text: str, candidate_texts: List[str], 
                         model: str = "embedding-3") -> List[Dict[str, Any]]:
        """找到与查询文本最相似的候选文本"""
        return self.embedding_client.find_most_similar(query_text, candidate_texts, model)

# 创建全局文本嵌入实例
embedding_generator = EmbeddingGenerator()

@mcp.tool()
def get_text_embeddings(
    input_text: Union[str, List[str]],
    model: str = "embedding-3"
) -> Dict[str, Any]:
    """
    获取文本的嵌入向量表示
    
    Args:
        input_text: 输入文本，可以是单个字符串或字符串列表
        model: 嵌入模型名称 (embedding-3, embedding-2)
    
    Returns:
        包含嵌入向量的结果字典
    """
    try:
        if not input_text:
            return {
                "success": False,
                "error": "输入文本不能为空"
            }
        
        result = embedding_generator.get_embeddings(
            input_text=input_text,
            model=model
        )
        
        return {
            "success": True,
            "model": result.get("model"),
            "data": result.get("data"),
            "usage": result.get("usage")
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"文本嵌入失败: {str(e)}"
        }

@mcp.tool()
def get_batch_embeddings(
    texts: List[str],
    model: str = "embedding-3"
) -> Dict[str, Any]:
    """
    批量获取多个文本的嵌入向量
    
    Args:
        texts: 文本列表
        model: 嵌入模型名称 (embedding-3, embedding-2)
    
    Returns:
        包含批量嵌入向量的结果字典
    """
    try:
        if not texts:
            return {
                "success": False,
                "error": "文本列表不能为空"
            }
        
        # 过滤空文本
        valid_texts = [text.strip() for text in texts if text and text.strip()]
        
        if not valid_texts:
            return {
                "success": False,
                "error": "没有找到有效的文本"
            }
        
        embeddings = embedding_generator.get_batch_embeddings(
            texts=valid_texts,
            model=model
        )
        
        return {
            "success": True,
            "model": model,
            "embeddings": embeddings,
            "count": len(embeddings),
            "dimension": len(embeddings[0]) if embeddings else 0
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"批量文本嵌入失败: {str(e)}"
        }

@mcp.tool()
def calculate_text_similarity(
    text1: str,
    text2: str,
    model: str = "embedding-3"
) -> Dict[str, Any]:
    """
    计算两个文本之间的语义相似度
    
    Args:
        text1: 第一个文本
        text2: 第二个文本
        model: 嵌入模型名称 (embedding-3, embedding-2)
    
    Returns:
        包含相似度分数的结果字典
    """
    try:
        if not text1 or not text2:
            return {
                "success": False,
                "error": "两个文本都不能为空"
            }
        
        # 获取两个文本的嵌入向量
        embedding1 = embedding_generator.get_single_embedding(text1, model)
        embedding2 = embedding_generator.get_single_embedding(text2, model)
        
        # 计算相似度
        similarity = embedding_generator.calculate_similarity(embedding1, embedding2)
        
        return {
            "success": True,
            "text1": text1,
            "text2": text2,
            "similarity": similarity,
            "model": model,
            "interpretation": {
                "score": similarity,
                "level": "高" if similarity > 0.8 else "中" if similarity > 0.5 else "低",
                "description": f"两个文本的语义相似度为 {similarity:.4f}"
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"相似度计算失败: {str(e)}"
        }

@mcp.tool()
def find_similar_texts(
    query_text: str,
    candidate_texts: List[str],
    model: str = "embedding-3",
    top_k: int = 5
) -> Dict[str, Any]:
    """
    在候选文本中找到与查询文本最相似的文本
    
    Args:
        query_text: 查询文本
        candidate_texts: 候选文本列表
        model: 嵌入模型名称 (embedding-3, embedding-2)
        top_k: 返回最相似的前k个结果
    
    Returns:
        包含相似文本排序结果的字典
    """
    try:
        if not query_text:
            return {
                "success": False,
                "error": "查询文本不能为空"
            }
        
        if not candidate_texts:
            return {
                "success": False,
                "error": "候选文本列表不能为空"
            }
        
        # 过滤空文本
        valid_candidates = [text.strip() for text in candidate_texts if text and text.strip()]
        
        if not valid_candidates:
            return {
                "success": False,
                "error": "没有找到有效的候选文本"
            }
        
        # 找到最相似的文本
        results = embedding_generator.find_most_similar(
            query_text=query_text,
            candidate_texts=valid_candidates,
            model=model
        )
        
        # 限制返回结果数量
        top_results = results[:min(top_k, len(results))]
        
        return {
            "success": True,
            "query": query_text,
            "model": model,
            "total_candidates": len(valid_candidates),
            "top_k": len(top_results),
            "results": top_results
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"相似文本搜索失败: {str(e)}"
        }

@mcp.tool()
def get_supported_embedding_models() -> Dict[str, Any]:
    """
    获取支持的嵌入模型列表
    
    Returns:
        包含支持模型的结果字典
    """
    try:
        models = embedding_client.get_available_models()
        
        return {
            "success": True,
            "models": models,
            "default_model": "embedding-3",
            "model_info": {
                "embedding-3": "最新的嵌入模型，提供高质量的文本向量表示",
                "embedding-2": "较早版本的嵌入模型，兼容性更好"
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"获取支持的模型失败: {str(e)}"
        }

@mcp.tool()
def test_embedding_api(test_text: Optional[str] = None) -> Dict[str, Any]:
    """
    测试文本嵌入API连接和功能
    
    Args:
        test_text: 可选的测试文本
    
    Returns:
        包含测试结果的字典
    """
    try:
        # 测试API连接
        connection_test = embedding_client.test_connection()
        
        result: Dict[str, Any] = {
            "success": True,
            "connection_test": connection_test,
            "supported_models": embedding_client.get_available_models()
        }
        
        # 如果提供了测试文本，进行嵌入测试
        if test_text:
            try:
                embedding = embedding_client.get_single_embedding(test_text, "embedding-3")
                result["embedding_test"] = {
                    "success": True,
                    "text": test_text,
                    "dimension": len(embedding),
                    "sample_values": embedding[:5] if len(embedding) >= 5 else embedding
                }
            except Exception as e:
                result["embedding_test"] = {
                    "success": False,
                    "error": str(e)
                }
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": f"API测试失败: {str(e)}"
        }

@mcp.tool()
def save_embeddings_to_file(
    texts: List[str],
    filename: str,
    model: str = "embedding-3"
) -> Dict[str, Any]:
    """
    将文本嵌入向量保存到文件
    
    Args:
        texts: 文本列表
        filename: 保存的文件名
        model: 嵌入模型名称
    
    Returns:
        保存结果字典
    """
    try:
        if not texts or not filename:
            return {
                "success": False,
                "error": "文本列表和文件名都是必需的"
            }
        
        # 获取嵌入向量
        embeddings = embedding_generator.get_batch_embeddings(texts, model)
        
        # 准备保存数据
        save_data = {
            "model": model,
            "timestamp": time.time(),
            "count": len(texts),
            "dimension": len(embeddings[0]) if embeddings else 0,
            "data": [
                {
                    "text": text,
                    "embedding": embedding
                }
                for text, embedding in zip(texts, embeddings)
            ]
        }
        
        # 创建唯一文件名
        file_id = str(uuid.uuid4())[:8]
        name, ext = os.path.splitext(filename)
        if not ext:
            ext = ".json"
        unique_filename = f"{name}_{file_id}{ext}"
        file_path = OUTPUTS_DIR / unique_filename
        
        # 保存文件
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        return {
            "success": True,
            "file_path": str(file_path),
            "filename": unique_filename,
            "size": file_path.stat().st_size,
            "count": len(texts),
            "dimension": save_data["dimension"],
            "model": model
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"保存嵌入向量失败: {str(e)}"
        }

@mcp.tool()
def load_embeddings_from_file(filename: str) -> Dict[str, Any]:
    """
    从文件加载文本嵌入向量
    
    Args:
        filename: 文件名
    
    Returns:
        加载结果字典
    """
    try:
        if not filename:
            return {
                "success": False,
                "error": "文件名不能为空"
            }
        
        # 检查文件是否存在
        file_path = Path(filename)
        if not file_path.exists():
            # 尝试在输出目录中查找
            output_path = OUTPUTS_DIR / file_path.name
            if output_path.exists():
                file_path = output_path
            else:
                return {
                    "success": False,
                    "error": f"文件不存在: {filename}"
                }
        
        # 加载文件
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return {
            "success": True,
            "filename": file_path.name,
            "model": data.get("model"),
            "count": data.get("count"),
            "dimension": data.get("dimension"),
            "timestamp": data.get("timestamp"),
            "data": data.get("data", [])
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"加载嵌入向量失败: {str(e)}"
        }

# 新增：知识库索引与搜索 MCP 工具
@mcp.tool()
def index_documents(
    paths: List[str],
    kb: str = "kb_default",
    chunk_size: int = 500,
    overlap: int = 50,
    model: str = "embedding-3"
) -> Dict[str, Any]:
    """
    索引本地/URL 文档到指定知识库(kb)
    - 支持 .md / .txt 文件与目录递归
    - 支持 http/https URL
    安全：本地路径受环境变量 ALLOW_INDEX_DIRS 控制（; 分隔的白名单），为空则不限制
    """
    try:
        if not paths or not isinstance(paths, list):
            return {"success": False, "error": "paths 不能为空且需为列表"}

        sources = collect_sources(paths)
        if not sources:
            return {"success": False, "error": "未找到可索引的文档"}

        def embed_batch_fn(texts: List[str], m: str):
            return embedding_generator.get_batch_embeddings(texts, m)

        stats = index_to_chroma(
            kb=kb,
            sources=sources,
            embed_batch_fn=embed_batch_fn,
            chunk_size=chunk_size,
            overlap=overlap,
            model=model,
        )
        return {"success": True, **stats}
    except Exception as e:
        return {"success": False, "error": f"索引失败: {str(e)}"}


@mcp.tool()
def semantic_search(
    query: str,
    kb: str = "kb_default",
    top_k: int = 5,
    model: str = "embedding-3"
) -> Dict[str, Any]:
    """
    在指定知识库(kb)中进行语义搜索，返回TopK片段
    """
    try:
        if not query:
            return {"success": False, "error": "query 不能为空"}

        def embed_single_fn(text: str, m: str):
            return embedding_generator.get_single_embedding(text, m)

        res = search_kb(
            kb=kb,
            query=query,
            embed_single_fn=embed_single_fn,
            top_k=top_k,
            model=model,
        )
        return res
    except Exception as e:
        return {"success": False, "error": f"搜索失败: {str(e)}"}


@mcp.tool()
def nl_command(command: str) -> Dict[str, Any]:
    """
    解析中文自然语言指令并路由：
    - 索引: 例 '索引 docs kb=kb_docs chunk_size=500 overlap=50'
    - 搜索: 例 '搜索 "差旅报销怎么走" kb=kb_docs top=5'
    """
    try:
        p = parse_command(command)
        action = p.get("action")
        if action == "index":
            return index_documents(
                paths=p.get("paths", []),
                kb=p.get("kb", "kb_default"),
                chunk_size=int(p.get("chunk_size", 500)),
                overlap=int(p.get("overlap", 50)),
                model=p.get("model", "embedding-3"),
            )
        if action == "search":
            return semantic_search(
                query=p.get("query", ""),
                kb=p.get("kb", "kb_default"),
                top_k=int(p.get("top", 5)),
                model=p.get("model", "embedding-3"),
            )
        return {"success": False, "error": p.get("error") or "无法识别的指令"}
    except Exception as e:
        return {"success": False, "error": f"指令执行失败: {str(e)}"}

def run_interactive_mode():
    """运行交互式文本嵌入模式"""
    print("=" * 60)
    print("🔤 AI文本嵌入生成器 - 交互模式")
    print("=" * 60)
    print("支持的功能:")
    print("1. 文本嵌入向量生成")
    print("2. 批量文本嵌入")
    print("3. 文本相似度计算")
    print("4. 相似文本搜索")
    print("5. 查看支持的模型")
    print("6. 测试API连接")
    print("7. 保存嵌入向量到文件")
    print("8. 从文件加载嵌入向量")
    print("9. 启动MCP服务器")
    print("0. 退出")
    print("=" * 60)
    
    while True:
        try:
            choice = input("\n请选择功能 (0-9): ").strip()
            
            if choice == "0":
                print("👋 再见!")
                break
            elif choice == "1":
                handle_text_embedding()
            elif choice == "2":
                handle_batch_embedding()
            elif choice == "3":
                handle_similarity_calculation()
            elif choice == "4":
                handle_similar_text_search()
            elif choice == "5":
                handle_model_info()
            elif choice == "6":
                handle_api_test()
            elif choice == "7":
                handle_save_embeddings()
            elif choice == "8":
                handle_load_embeddings()
            elif choice == "9":
                print("🔧 启动MCP服务器...")
                mcp.run(transport="sse")
                break
            else:
                print("❌ 无效选择，请输入0-9")
                
        except KeyboardInterrupt:
            print("\n👋 再见!")
            break
        except Exception as e:
            print(f"❌ 错误: {e}")

def handle_text_embedding():
    """处理文本嵌入"""
    print("\n🔤 文本嵌入向量生成")
    text = input("请输入要生成嵌入向量的文本: ").strip()
    if not text:
        print("❌ 文本不能为空")
        return
    
    models = embedding_client.get_available_models()
    print(f"\n可用的模型: {', '.join(models)}")
    model = input("请选择模型 (默认: embedding-3): ").strip() or "embedding-3"
    
    print("🔍 生成中...")
    try:
        result = get_text_embeddings(text, model)
        
        if result["success"]:
            data = result["data"][0] if result["data"] else {}
            embedding = data.get("embedding", [])
            
            print(f"✅ 嵌入向量生成成功!")
            print(f"文本: {text}")
            print(f"模型: {result['model']}")
            print(f"向量维度: {len(embedding)}")
            print(f"前5个值: {embedding[:5]}")
            print(f"使用情况: {result.get('usage', {})}")
        else:
            print(f"❌ 生成失败: {result['error']}")
    except Exception as e:
        print(f"❌ 生成失败: {str(e)}")

def handle_batch_embedding():
    """处理批量文本嵌入"""
    print("\n📁 批量文本嵌入")
    print("请输入要生成嵌入向量的文本 (每行一个，空行结束):")
    
    texts = []
    while True:
        text = input().strip()
        if not text:
            break
        texts.append(text)
    
    if not texts:
        print("❌ 没有输入任何文本")
        return
    
    models = embedding_client.get_available_models()
    print(f"\n可用的模型: {', '.join(models)}")
    model = input("请选择模型 (默认: embedding-3): ").strip() or "embedding-3"
    
    print(f"🔍 批量生成 {len(texts)} 个文本的嵌入向量...")
    try:
        result = get_batch_embeddings(texts, model)
        
        if result["success"]:
            print(f"✅ 批量嵌入生成成功!")
            print(f"文本数量: {result['count']}")
            print(f"向量维度: {result['dimension']}")
            print(f"模型: {result['model']}")
        else:
            print(f"❌ 批量生成失败: {result['error']}")
    except Exception as e:
        print(f"❌ 批量生成失败: {str(e)}")

def handle_similarity_calculation():
    """处理相似度计算"""
    print("\n🔍 文本相似度计算")
    text1 = input("请输入第一个文本: ").strip()
    text2 = input("请输入第二个文本: ").strip()
    
    if not text1 or not text2:
        print("❌ 两个文本都不能为空")
        return
    
    models = embedding_client.get_available_models()
    print(f"\n可用的模型: {', '.join(models)}")
    model = input("请选择模型 (默认: embedding-3): ").strip() or "embedding-3"
    
    print("🔍 计算中...")
    try:
        result = calculate_text_similarity(text1, text2, model)
        
        if result["success"]:
            print(f"✅ 相似度计算成功!")
            print(f"文本1: {result['text1']}")
            print(f"文本2: {result['text2']}")
            print(f"相似度: {result['similarity']:.4f}")
            print(f"相似度等级: {result['interpretation']['level']}")
            print(f"描述: {result['interpretation']['description']}")
        else:
            print(f"❌ 计算失败: {result['error']}")
    except Exception as e:
        print(f"❌ 计算失败: {str(e)}")

def handle_similar_text_search():
    """处理相似文本搜索"""
    print("\n🔍 相似文本搜索")
    query = input("请输入查询文本: ").strip()
    if not query:
        print("❌ 查询文本不能为空")
        return
    
    print("请输入候选文本 (每行一个，空行结束):")
    candidates = []
    while True:
        text = input().strip()
        if not text:
            break
        candidates.append(text)
    
    if not candidates:
        print("❌ 没有输入任何候选文本")
        return
    
    models = embedding_client.get_available_models()
    print(f"\n可用的模型: {', '.join(models)}")
    model = input("请选择模型 (默认: embedding-3): ").strip() or "embedding-3"
    
    top_k = input("请输入返回结果数量 (默认: 5): ").strip()
    try:
        top_k = int(top_k) if top_k else 5
    except ValueError:
        top_k = 5
    
    print("🔍 搜索中...")
    try:
        result = find_similar_texts(query, candidates, model, top_k)
        
        if result["success"]:
            print(f"✅ 相似文本搜索成功!")
            print(f"查询: {result['query']}")
            print(f"候选文本数量: {result['total_candidates']}")
            print(f"返回结果数量: {result['top_k']}")
            print("\n最相似的文本:")
            for i, item in enumerate(result['results']):
                print(f"{i+1}. {item['text']} (相似度: {item['similarity']:.4f})")
        else:
            print(f"❌ 搜索失败: {result['error']}")
    except Exception as e:
        print(f"❌ 搜索失败: {str(e)}")

def handle_model_info():
    """处理模型信息查看"""
    print("\n🔧 支持的嵌入模型")
    try:
        result = get_supported_embedding_models()
        
        if result["success"]:
            print("✅ 可用的模型:")
            for model in result["models"]:
                info = result["model_info"].get(model, "无描述")
                print(f"  {model}: {info}")
            print(f"\n默认模型: {result['default_model']}")
        else:
            print(f"❌ 获取模型信息失败: {result['error']}")
    except Exception as e:
        print(f"❌ 获取模型信息失败: {str(e)}")

def handle_api_test():
    """处理API测试"""
    print("\n🔧 API连接测试")
    test_text = input("请输入测试文本 (可选): ").strip() or None
    
    print("🔍 测试中...")
    try:
        result = test_embedding_api(test_text)
        
        if result["success"]:
            print("✅ API测试结果:")
            conn_test = result["connection_test"]
            print(f"  连接状态: {'正常' if conn_test else '失败'}")
            print(f"  支持的模型: {', '.join(result['supported_models'])}")
            
            if 'embedding_test' in result:
                embed_test = result['embedding_test']
                if embed_test['success']:
                    print(f"  测试嵌入: 成功生成 {embed_test['dimension']} 维向量")
                    print(f"  示例值: {embed_test['sample_values']}")
                else:
                    print(f"  测试嵌入失败: {embed_test['error']}")
        else:
            print(f"❌ API测试失败: {result['error']}")
    except Exception as e:
        print(f"❌ API测试失败: {str(e)}")

def handle_save_embeddings():
    """处理保存嵌入向量"""
    print("\n💾 保存嵌入向量到文件")
    print("请输入要保存的文本 (每行一个，空行结束):")
    
    texts = []
    while True:
        text = input().strip()
        if not text:
            break
        texts.append(text)
    
    if not texts:
        print("❌ 没有输入任何文本")
        return
    
    filename = input("请输入文件名: ").strip()
    if not filename:
        print("❌ 文件名不能为空")
        return
    
    models = embedding_client.get_available_models()
    print(f"\n可用的模型: {', '.join(models)}")
    model = input("请选择模型 (默认: embedding-3): ").strip() or "embedding-3"
    
    print("💾 保存中...")
    try:
        result = save_embeddings_to_file(texts, filename, model)
        
        if result["success"]:
            print(f"✅ 保存成功!")
            print(f"文件路径: {result['file_path']}")
            print(f"文件大小: {result['size']} 字节")
            print(f"文本数量: {result['count']}")
            print(f"向量维度: {result['dimension']}")
        else:
            print(f"❌ 保存失败: {result['error']}")
    except Exception as e:
        print(f"❌ 保存失败: {str(e)}")

def handle_load_embeddings():
    """处理加载嵌入向量"""
    print("\n📂 从文件加载嵌入向量")
    filename = input("请输入文件名: ").strip()
    if not filename:
        print("❌ 文件名不能为空")
        return
    
    print("📂 加载中...")
    try:
        result = load_embeddings_from_file(filename)
        
        if result["success"]:
            print(f"✅ 加载成功!")
            print(f"文件名: {result['filename']}")
            print(f"模型: {result['model']}")
            print(f"文本数量: {result['count']}")
            print(f"向量维度: {result['dimension']}")
            
            # 显示前几个文本
            data = result['data']
            print("\n前3个文本:")
            for i, item in enumerate(data[:3]):
                print(f"{i+1}. {item['text'][:50]}...")
        else:
            print(f"❌ 加载失败: {result['error']}")
    except Exception as e:
        print(f"❌ 加载失败: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--mcp":
            print("🔧 启动MCP服务器模式...")
            mcp.run(transport="sse")
        elif sys.argv[1] == "--test":
            print("🧪 运行测试...")
            # 可以添加测试代码
            handle_api_test()
        else:
            print("❌ 未知参数，支持的参数: --mcp, --test")
    else:
        # 默认运行交互式模式
        run_interactive_mode()