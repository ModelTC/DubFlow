"""Video dubbing CLI; stage orchestration lives in runtime.pipeline."""

import argparse

from runtime.config import DEFAULT_CONFIG
from runtime.languages import LANGUAGES
from runtime.pipeline import run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source", required=True, help="原始视频，包含原语言对白")
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--merge-speaker-turns", action=argparse.BooleanOptionalAction, default=None,
                        help="合并同一人连续发言；不跨换人、重叠或切镜，默认沿用配置")
    parser.add_argument("--context-seconds", type=float, help="覆盖 TBDub 输入视频前后上下文秒数；贴回上下文不会超过此值")
    parser.add_argument("--transition-seconds", type=float, help="覆盖原片与 TBDub 画面的贴回淡变宽度（秒）")
    parser.add_argument("--analysis-plan", help="复用相同原片的说话人/人脸分析计划")
    parser.add_argument("--source-audio-manifest", help="复用相同原片分离出的对白和背景")
    parser.add_argument("--dialogue", help="复用相同分析计划已完成的翻译/配音结果")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--gpu", type=int, help="覆盖配置中的单卡编号")
    parser.add_argument("--source-language", choices=("auto", *LANGUAGES))
    parser.add_argument("--target-language", choices=tuple(LANGUAGES), required=True,
                        help="必填：输出对白语言")
    parser.add_argument("--duration-mode", choices=("fixed",), default="fixed",
                        help="保留原视频时间线（默认）")
    run(parser.parse_args())
