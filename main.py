"""Command-line entry point.

Example:
    python main.py --source videos/CAR.mp4
"""
import argparse

from src.pipeline import run


def parse_args():
    parser = argparse.ArgumentParser(description='Motorcycle collision warning system')
    parser.add_argument('--source', default='videos/CAR.mp4', help='input video')
    parser.add_argument('--config', default=None, help='config file (default: config.yaml)')
    parser.add_argument('--device', default=None, help="'' for auto, 'cpu', or a CUDA index such as '0'")
    parser.add_argument('--skip-lanes', action='store_true', help='reuse existing lane data if available')
    parser.add_argument('--no-lane-video', action='store_true', help='do not save the lane detection video')
    parser.add_argument('--no-audio', action='store_true', help='do not add warning sounds')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args.source, config_path=args.config, skip_lanes=args.skip_lanes,
        no_audio=args.no_audio, save_lane_video=not args.no_lane_video, device=args.device)
