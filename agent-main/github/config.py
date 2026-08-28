import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """系统核心配置类"""
    #API配置
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com")
    DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-m3")

    # 大模型生成参数
    TEMPERATURE_INTENT = float(os.getenv("TEMPERATURE_INTENT", 0.1))
    TEMPERATURE_REPLY = float(os.getenv("TEMPERATURE_REPLY", 0.7))

    # 对话管理配置
    MAX_CONTEXT_LENGTH = int(os.getenv("MAX_CONTEXT_LENGTH", 20))

    # RAG配置
    CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 200))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 20))
    TOP_K = int(os.getenv("TOP_K", 3))

    # 意图列表
    ALLOWED_INTENTS = ["查订单", "退款", "物流", "售后", "其他"]


# 单例配置实例
config = Config()


# 配置校验
def validate_config():
    if not config.DEEPSEEK_API_KEY:
        raise ValueError("请配置DEEPSEEK_API_KEY环境变量")
    # 创建向量库目录
    if not os.path.exists(config.CHROMA_PERSIST_DIRECTORY):
        os.makedirs(config.CHROMA_PERSIST_DIRECTORY)
    return True