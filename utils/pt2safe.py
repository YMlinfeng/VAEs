import os
import argparse
import torch
from safetensors.torch import save_file

def convert_pt_to_safetensors(input_path: str, output_path: str = None):
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    # 自动设置输出路径（如果是目录就加上文件名）
    if output_path is None:
        output_path = os.path.splitext(input_path)[0] + ".safetensors"
    elif os.path.isdir(output_path):
        # 如果是目录，则自动拼接输入文件名
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_path, base_name + ".safetensors")

    print(f"Loading checkpoint from: {input_path}")
    state = torch.load(input_path, map_location="cpu")

    # 如果是 PyTorch Lightning 格式
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # # 去掉可能存在的 "module." 前缀（DataParallel）
    # new_state = {
    #     k.replace("module.", ""): v for k, v in state.items()
    # }

    print(f"Saving to safetensors format: {output_path}")
    save_file(state, output_path)
    print("✅ Conversion complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .pt/.pth to .safetensors")
    parser.add_argument(
        "--input",
        default="/mnt/bn/occupancy3d/workspace/mzj/Open-Sora-Plan/baseline/wan/Wan2.1_VAE.pth",
        help="Path to the input .pt or .pth file"
    )
    parser.add_argument(
        "--output",
        default="/mnt/bn/occupancy3d/workspace/mzj/Open-Sora-Plan/baseline/wan",
        help="Path to output .safetensors file or directory"
    )
    args = parser.parse_args()

    convert_pt_to_safetensors(args.input, args.output)


# 在 Python 的 argparse 中：

# 位置参数（例如 input）是必须提供的，不需要加 --。
# 可选参数（例如 --output）使用双横线 -- 开头，可以选择性提供。