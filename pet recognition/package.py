"""
打包脚本 — 使用 PyInstaller 将猫品种分类系统打包为可执行文件

用法:
    python package.py

打包完成后在 dist/CatClassifier/ 目录下生成完整应用，
双击 CatClassifier.exe 即可运行。

优化策略:
  - CPU-only PyTorch（省去 3.7GB CUDA 库）
  - 排除 scipy/matplotlib/pandas/seaborn/sklearn 等训练用依赖
  - --onedir 文件夹模式（启动快）
"""

import subprocess
import sys
import os
import importlib
import shutil


def _find_catcls_python():
    """在 cat-cls 环境中查找 Python 解释器"""
    # conda envs 常见路径
    candidates = [
        os.path.join(os.path.expanduser("~"), "anaconda3", "envs", "cat-cls", "python.exe"),
        os.path.join(os.path.expanduser("~"), "Anaconda3", "envs", "cat-cls", "python.exe"),
        "C:/Users/TANG/anaconda3/envs/cat-cls/python.exe",
        "C:/ProgramData/anaconda3/envs/cat-cls/python.exe",
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def _check_env(python_exe):
    """检查指定 Python 环境是否具备打包所需的全部依赖"""
    try:
        r = subprocess.run(
            [python_exe, "-c",
             "import PyInstaller, torch, torchvision, PIL, numpy"],
            capture_output=True, text=True, timeout=30
        )
        return r.returncode == 0
    except Exception:
        return False


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dist_name = "CatClassifier"

    # ── 自动选择最优打包环境 ──────────────────────────────────
    # 优先使用当前 Python，如果不可用则切换到 cat-cls 环境
    pack_python = sys.executable

    if not _check_env(pack_python):
        print("⚠️  当前 Python 环境缺少 PyInstaller/torch 等依赖")
        print("   正在查找 cat-cls 轻量打包环境...")
        catcls = _find_catcls_python()
        if catcls and _check_env(catcls):
            pack_python = catcls
            print(f"✅  已切换到: {catcls}")
        else:
            print("\n❌  未找到可用的打包环境 (cat-cls)")
            print("   请先创建环境: conda create -n cat-cls python=3.8 -y")
            print("   并安装: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
            print("           pip install Pillow numpy pyinstaller")
            return
    else:
        # 当前环境可用，但检查是否是 CPU-only torch（避免超大体积）
        r = subprocess.run(
            [pack_python, "-c", "import torch; print(torch.cuda.is_available())"],
            capture_output=True, text=True, timeout=10
        )
        if "True" in r.stdout:
            print("⚠️  当前环境为 CUDA 版 PyTorch，打包体积会很大 (~4.5GB)")
            print("   建议切换到 cat-cls 环境 (CPU-only, ~570MB)")
            print("   继续使用当前环境按 Enter，或 Ctrl+C 取消...\n")
            # 不阻塞，只提醒
    print()

    # ── 先检查关键文件 ────────────────────────────────────────
    weights_path = os.path.join(base_dir, "outputs", "ResNet50", "best.pth")
    if not os.path.isfile(weights_path):
        print("⚠️   未找到 outputs/ResNet50/best.pth")
        print("     请先运行 train.py 训练模型后再打包")
        print("     或手动放置权重文件到 outputs/ResNet50/ 目录")
        return  # 提前退出，保护已有 dist/

    # ── 清理 PyInstaller 临时文件（不碰 outputs/ 和已有 dist/）──
    build_path = os.path.join(base_dir, "build")
    if os.path.isdir(build_path):
        shutil.rmtree(build_path)
        print(f"🧹  清理: build/")
    spec_file = os.path.join(base_dir, f"{dist_name}.spec")
    if os.path.isfile(spec_file):
        os.remove(spec_file)

    dist_path = os.path.join(base_dir, "dist", dist_name)
    if os.path.isdir(dist_path):
        print(f"⚠️   dist/{dist_name}/ 已存在，将被覆盖")

    # ── 组装 PyInstaller 命令 ──────────────────────────────────
    cmd = [
        pack_python, "-m", "PyInstaller",
        "--noconfirm",
        "--onedir",                          # 文件夹模式
        "--windowed",                        # 不显示控制台
        "--clean",                           # 清理 PyInstaller 缓存
        "--name", dist_name,

        # ── 排除主程序用不到的训练用库（省体积）─────────────
        "--exclude-module", "scipy",
        "--exclude-module", "matplotlib",
        "--exclude-module", "pandas",
        "--exclude-module", "seaborn",
        "--exclude-module", "sklearn",
        "--exclude-module", "IPython",
        "--exclude-module", "jupyter",
        "--exclude-module", "notebook",
        "--exclude-module", "tensorboard",
        "--exclude-module", "torch.utils.tensorboard",

        # ── 隐式导入（PyTorch / tkinter）──────────────────────
        # PyTorch 动态加载的子模块 — 必须显式列出
        "--hidden-import", "torch",
        "--hidden-import", "torch.nn",
        "--hidden-import", "torch.distributed",
        "--hidden-import", "torch.distributed.rpc",
        "--hidden-import", "torch.distributed.distributed_c10d",
        "--hidden-import", "torch._C",
        "--hidden-import", "torch._C._distributed_c10d",
        "--hidden-import", "torch.cuda",
        "--hidden-import", "torchvision",
        "--hidden-import", "torchvision.transforms",
        "--hidden-import", "torchvision.models",
        "--hidden-import", "torchvision.transforms.functional",
        "--hidden-import", "PIL._tkinter_finder",
        "--hidden-import", "PIL.Image",
        "--hidden-import", "numpy",
    ]

    # 携带 icon
    icon_path = os.path.join(base_dir, "icon.ico")
    if os.path.isfile(icon_path):
        cmd += ["--icon", icon_path]

    # 携带 README
    readme_path = os.path.join(base_dir, "README.md")
    if os.path.isfile(readme_path):
        cmd += ["--add-data", f"{readme_path};."]

    # 携带模型权重（核心，已在入口处校验存在）
    cmd += ["--add-data", f"{weights_path};outputs/ResNet50"]
    print(f"✅  将携带模型权重: best.pth")

    # 主程序入口
    cmd.append(os.path.join(base_dir, "cat_classifier.py"))

    print("=" * 60)
    print("  🐱  宠物猫品种分类系统  (CPU 轻量版)")
    print("=" * 60)
    print(f"Python: {pack_python}")
    print(f"工作目录: {base_dir}")
    print("优化: CPU-only PyTorch + 排除训练用依赖\n")
    print("正在打包，请稍候...\n")

    result = subprocess.run(cmd, cwd=base_dir)

    if result.returncode == 0:
        print("\n" + "=" * 60)
        print("  ✅  打包成功！")
        print("=" * 60)
        print(f"\n📁  应用目录: {dist_path}")
        print(f"🚀  可执行文件: {os.path.join(dist_path, dist_name + '.exe')}")

        # 复制权重到 dist 的应用目录
        dest_weights_dir = os.path.join(dist_path, "outputs", "ResNet50")
        os.makedirs(dest_weights_dir, exist_ok=True)
        dest_weights = os.path.join(dest_weights_dir, "best.pth")
        if not os.path.isfile(dest_weights):
            shutil.copy2(weights_path, dest_weights)
            print("✅  已复制模型权重到应用目录")

        # ── 修复 Anaconda DLL 依赖缺失 ──────────────────────
        # Anaconda python38.dll 依赖 Library/bin 下的 VC++ 运行时 DLL，
        # PyInstaller 可能遗漏，手动补齐
        _internal = os.path.join(dist_path, "_internal")
        anaconda_lib_bin = os.path.join(os.path.dirname(pack_python), "Library", "bin")
        missing_dlls = [
            "concrt140.dll", "msvcp140_1.dll", "msvcp140_2.dll",
            "msvcp140_atomic_wait.dll", "msvcp140_codecvt_ids.dll",
            "vcruntime140_threads.dll",
        ]
        copied_dlls = 0
        for dll in missing_dlls:
            src = os.path.join(anaconda_lib_bin, dll)
            dst = os.path.join(_internal, dll)
            if os.path.isfile(src) and not os.path.isfile(dst):
                shutil.copy2(src, dst)
                copied_dlls += 1
        if copied_dlls:
            print(f"✅  已补全 VC++ 运行时 DLL ({copied_dlls} 个)")

        # 生成启动说明
        launch_txt = os.path.join(dist_path, "启动说明.txt")
        with open(launch_txt, "w", encoding="utf-8") as f:
            f.write("宠物猫品种智能分类系统 v1.0 (CPU 轻量版)\n")
            f.write("=" * 45 + "\n\n")
            f.write("【启动方式】\n")
            f.write("  双击 CatClassifier.exe 即可运行\n\n")
            f.write("【首次使用】\n")
            f.write("  1. 程序启动后自动加载 ResNet50 模型\n")
            f.write("  2. 点击「打开图片」选择猫咪照片\n")
            f.write("  3. 点击「开始识别」进行品种识别\n\n")
            f.write("【支持品种（12种）】\n")
            breeds = [
                "俄罗斯蓝猫", "加拿大无毛猫(斯芬克斯)", "埃及猫", "孟买猫",
                "孟加拉猫", "布偶猫", "暹罗猫", "波斯猫",
                "缅因猫", "缅甸猫", "英国短毛猫", "阿比西尼亚猫"
            ]
            for b in breeds:
                f.write(f"  · {b}\n")
            f.write("\n【运行要求】\n")
            f.write("  · 需要 Windows 10/11 64位系统\n")
            f.write("  · CPU 推理（无需独立显卡）\n")
            f.write("  · 请勿删除 _internal 文件夹\n")
            f.write("\n【更换权重】\n")
            f.write("  菜单「文件 → 加载模型权重」可加载自定义 .pth 文件\n")

        print(f"📄  已生成启动说明: {launch_txt}")
        print(f"\n💡  将整个 dist/{dist_name}/ 文件夹复制到任意位置均可使用")

    else:
        print(f"\n❌  打包失败 (返回码: {result.returncode})")
        print("提示: pip install pyinstaller --upgrade")


if __name__ == "__main__":
    main()
