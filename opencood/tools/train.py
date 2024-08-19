import argparse
import sys

from icecream import ic

from opencood.utils.file_utils import load_yaml



def args_parser():
    """
    这段函数可以直接从命令行中读取参数, 而无需传参
    Returns:
    解析的命令行参数
    """
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--config", type=str, required=True, help='data generation yaml file needed')
    parser.add_argument('--model_dir', default='', help='Continued training path')
    parser.add_argument('--fusion_method', '-f', default="intermediate", help='passed to inference.')
    args = parser.parse_args()
    return args


def main():
    args = args_parser()
    configs = load_yaml(args.config, args.model_dir)
    ic(configs)  # 测试专用


if __name__ == '__main__':
    sys.argv = ["train.py", "--config", "../../opencood/configs/OPV2V/MoreModality/HEAL/stage1/m1_pyramid.yaml"]
    main()
