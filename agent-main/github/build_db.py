import os
from knowledge_base import KnowledgeBase
from config import config

def build_database():
    """
    构建/更新向量数据库
    从 data/ 文件夹读取所有 .txt 文档，增量添加到向量库中
    """
    print("=" * 60)
    print("🚀 开始构建知识库向量...")
    print("=" * 60)

    # 1. 初始化知识库（会自动加载已有的向量库或创建空库）
    kb = KnowledgeBase()

    # 2. 检查数据目录
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"📁 已创建 {data_dir}/ 文件夹，请将您的 .txt 文档放入其中")
        return

    # 3. 获取所有 .txt 文件
    file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.txt')]
    if not file_paths:
        print(f"⚠️ 未在 {data_dir}/ 找到任何 .txt 文件")
        return

    print(f"📄 发现以下文档：")
    for fp in file_paths:
        print(f"   - {fp}")

    # 4. 执行添加（增量更新，不会覆盖旧数据）
    success = kb.add_knowledge_doc(file_paths)

    # 5. 结果反馈
    if success:
        print(f"✅ 向量库构建成功！数据保存在：{config.CHROMA_PERSIST_DIRECTORY}")
    else:
        print("❌ 构建失败，请查看日志排查错误。")
    print("=" * 60)

if __name__ == "__main__":
    build_database()