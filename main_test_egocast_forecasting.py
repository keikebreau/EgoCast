import argparse
import logging
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.select_dataset import define_Dataset
from models.select_model import define_Model
from utils import utils_logger
from utils import utils_option as option


def _as_float(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-opt",
        type=str,
        default="options/test_egocast_forecasting.json",
        help="Path to the forecasting test option JSON file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path. Overrides path.pretrained_netG in the option file.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on the number of test sequences to evaluate.",
    )
    parser.add_argument(
        "--skip-indices",
        type=int,
        nargs="*",
        default=[17, 62],
        help="Test indices to skip. Defaults match main_train_egocast.py.",
    )
    pred_group = parser.add_mutually_exclusive_group()
    pred_group.add_argument(
        "--gt-input",
        action="store_true",
        help="Use ground-truth past skeletons as forecasting input by forcing pred_input=false.",
    )
    pred_group.add_argument(
        "--pred-input",
        action="store_true",
        help="Use predicted current-frame skeletons as forecasting input by forcing pred_input=true.",
    )
    args = parser.parse_args()

    opt = option.parse(args.opt, is_train=False)

    if args.checkpoint is not None:
        opt["path"]["pretrained_netG"] = args.checkpoint
        opt["path"]["pretrained"] = args.checkpoint

    if args.gt_input:
        opt["datasets"]["test"]["pred_input"] = False
    elif args.pred_input:
        opt["datasets"]["test"]["pred_input"] = True

    if not opt["datasets"]["test"]["future"]:
        raise ValueError("This script is only for forecasting configs with datasets.test.future=true.")

    if opt["datasets"]["test"]["dataloader_batch_size"] != 1:
        raise ValueError("model.test_fcast() expects datasets.test.dataloader_batch_size to be 1.")

    if opt["datasets"]["test"].get("pred_input"):
        route = opt["datasets"]["test"].get("route")
        if route is None or not os.path.exists(route):
            raise FileNotFoundError(
                "datasets.test.pred_input=true requires datasets.test.route to point to "
                "current-frame prediction files. Use --gt-input to evaluate from ground-truth "
                "history instead."
            )

    for key in ["log", "images"]:
        path = opt["path"].get(key)
        if path is not None:
            os.makedirs(path, exist_ok=True)

    logger_name = "test_forecasting"
    utils_logger.logger_info(logger_name, os.path.join(opt["path"]["log"], logger_name + ".log"))
    logger = logging.getLogger(logger_name)
    logger.info(option.dict2str(opt))

    opt = option.dict_to_nonedict(opt)

    test_set = define_Dataset(opt["datasets"]["test"])
    test_loader = DataLoader(
        test_set,
        batch_size=opt["datasets"]["test"]["dataloader_batch_size"],
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=True,
    )

    model = define_Model(opt)
    model.init_test()

    pos_errors = []
    vel_errors = []
    aria_errors = []
    rot_errors = []
    skipped = set(args.skip_indices or [])

    with torch.no_grad():
        for index, test_data in enumerate(tqdm(test_loader, desc="Testing")):
            if args.max_samples is not None and len(pos_errors) >= args.max_samples:
                break
            if index in skipped:
                continue

            model.feed_data(test_data, test=True)
            result = model.test_fcast()

            if opt["datasets"]["test"]["output"] == "aria":
                pos_error, rot_error, vel_error = result[:3]
                rot_errors.append(_as_float(rot_error))
            else:
                pos_error, vel_error, aria_error = result[:3]
                aria_errors.append(_as_float(aria_error))

            pos_errors.append(_as_float(pos_error))
            vel_errors.append(_as_float(vel_error))

    if not pos_errors:
        raise RuntimeError("No test samples were evaluated.")

    mean_pos = sum(pos_errors) / len(pos_errors)
    mean_vel = sum(vel_errors) / len(vel_errors)

    if opt["datasets"]["test"]["output"] == "aria":
        mean_rot = sum(rot_errors) / len(rot_errors)
        message = (
            "Evaluated {:d} samples. Average positional error [cm]: {:.5f}, "
            "Average rotational error: {:.5f}, Average velocity error [cm/s]: {:.5f}"
        ).format(len(pos_errors), mean_pos * 100, mean_rot, mean_vel * 100)
    else:
        mean_aria = sum(aria_errors) / len(aria_errors)
        message = (
            "Evaluated {:d} samples. Average positional error [cm]: {:.5f}, "
            "Average velocity error [cm/s]: {:.5f}, Average aria error [cm]: {:.5f}"
        ).format(len(pos_errors), mean_pos * 100, mean_vel * 100, mean_aria * 100)

    logger.info(message)
    print(message)


if __name__ == "__main__":
    main()
