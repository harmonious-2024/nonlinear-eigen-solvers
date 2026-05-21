import json

# 1. 读取你的笔记本文件
file_path = "lab_ui.ipynb"
with open(file_path, "r", encoding="utf-8") as f:
    notebook = json.load(f)

# 2. 强行把所有单元格的 outputs（输出缓存）全部清空
for cell in notebook.get("cells", []):
    if "outputs" in cell:
        cell["outputs"] = []
    if "metadata" in cell and "execution" in cell["metadata"]:
        cell["metadata"].pop("execution")

# 3. 把洗干净的文件重新保存
with open(file_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("✨ 降维打击成功！ lab_ui.ipynb 已被彻底洗净！")