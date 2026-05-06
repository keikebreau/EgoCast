import argparse
import csv
import logging
import os
import re

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


def _new_curve_accumulators(future_frames):
    return {
        "pos_sum": torch.zeros(future_frames, dtype=torch.float64),
        "pos_count": torch.zeros(future_frames, dtype=torch.float64),
        "aria_sum": torch.zeros(future_frames, dtype=torch.float64),
        "aria_count": torch.zeros(future_frames, dtype=torch.float64),
        "vel_sum": torch.zeros(max(future_frames - 1, 0), dtype=torch.float64),
        "vel_count": torch.zeros(max(future_frames - 1, 0), dtype=torch.float64),
    }


def _add_curve_details(acc, details, output):
    if output == "aria":
        gt_aria = details["gt_aria"].detach().cpu()
        pred_aria = details["pred_aria"].detach().cpu()
        pos_error = torch.sqrt(torch.sum(torch.square(gt_aria[:, :, :3] - pred_aria[:, :, :3]), dim=-1))
        acc["pos_sum"] += pos_error.sum(dim=0).double()
        acc["pos_count"] += torch.full_like(acc["pos_count"], pos_error.shape[0], dtype=torch.float64)
        aria_error = pos_error
    else:
        gt_skeleton = details["gt_skeleton"].detach().cpu()
        pred_skeleton = details["pred_skeleton"].detach().cpu()
        visible = details["visible"].detach().cpu()
        joint_error = torch.sqrt(torch.sum(torch.square(gt_skeleton - pred_skeleton), dim=-1))
        masked_joint_error = joint_error * visible
        acc["pos_sum"] += masked_joint_error.sum(dim=(0, 2)).double()
        acc["pos_count"] += visible.sum(dim=(0, 2)).double()

        gt_aria = details["gt_aria"].detach().cpu()
        pred_aria = details["pred_aria"].detach().cpu()
        aria_error = torch.sqrt(torch.sum(torch.square(gt_aria[:, :, :3] - pred_aria[:, :, :3]), dim=-1))
        acc["aria_sum"] += aria_error.sum(dim=0).double()
        acc["aria_count"] += torch.full_like(acc["aria_count"], aria_error.shape[0], dtype=torch.float64)

        if gt_skeleton.shape[1] > 1:
            gt_velocity = (gt_skeleton[:, 1:, ...] - gt_skeleton[:, :-1, ...]) * 30
            pred_velocity = (pred_skeleton[:, 1:, ...] - pred_skeleton[:, :-1, ...]) * 30
            velocity_mask = visible[:, 1:, :] * visible[:, :-1, :]
            velocity_error = torch.sqrt(torch.sum(torch.square(gt_velocity - pred_velocity), dim=-1))
            masked_velocity_error = velocity_error * velocity_mask
            acc["vel_sum"] += masked_velocity_error.sum(dim=(0, 2)).double()
            acc["vel_count"] += velocity_mask.sum(dim=(0, 2)).double()
        return

    acc["aria_sum"] += aria_error.sum(dim=0).double()
    acc["aria_count"] += torch.full_like(acc["aria_count"], aria_error.shape[0], dtype=torch.float64)
    if gt_aria.shape[1] > 1:
        gt_velocity = (gt_aria[:, 1:, :3] - gt_aria[:, :-1, :3]) * 30
        pred_velocity = (pred_aria[:, 1:, :3] - pred_aria[:, :-1, :3]) * 30
        velocity_error = torch.sqrt(torch.sum(torch.square(gt_velocity - pred_velocity), dim=-1))
        acc["vel_sum"] += velocity_error.sum(dim=0).double()
        acc["vel_count"] += torch.full_like(acc["vel_count"], velocity_error.shape[0], dtype=torch.float64)


def _mean_or_none(sum_tensor, count_tensor, index):
    if index < 0 or index >= len(sum_tensor) or count_tensor[index] <= 0:
        return None
    return (sum_tensor[index] / count_tensor[index]).item()


def _write_curve_csv(path, acc):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "forecast_step",
            "mpjpe_cm",
            "aria_error_cm",
            "within_horizon_velocity_error_cm_s",
        ])
        for step in range(len(acc["pos_sum"])):
            pos = _mean_or_none(acc["pos_sum"], acc["pos_count"], step)
            aria = _mean_or_none(acc["aria_sum"], acc["aria_count"], step)
            vel = _mean_or_none(acc["vel_sum"], acc["vel_count"], step - 1)
            writer.writerow([
                step + 1,
                "" if pos is None else "{:.6f}".format(pos * 100),
                "" if aria is None else "{:.6f}".format(aria * 100),
                "" if vel is None else "{:.6f}".format(vel * 100),
            ])


def _forecast_output_dim(opt):
    dataset_opt = opt["datasets"]["train"]
    if dataset_opt["use_aria"]:
        if dataset_opt["use_rot"]:
            per_frame = 7 if dataset_opt["output"] == "aria" else 58
        else:
            per_frame = 3 if dataset_opt["output"] == "aria" else 54
    else:
        per_frame = 63
    return per_frame * dataset_opt["future_frames"]


def _slowfast_summary_dim(opt):
    net_opt = opt["netG"]
    embed_dim = net_opt["embed_dim"]
    beta = max(1, int(net_opt.get("slowfast_beta", 4)))
    fast_dim = max(8, embed_dim // beta)
    recent_frames = max(0, int(net_opt.get("slowfast_recent_frames", 0)))
    recent_dim = embed_dim * recent_frames
    if net_opt.get("slowfast_use_recent_delta", False):
        recent_dim += embed_dim
    video_dim = 768 if net_opt.get("video_model", False) else 0
    return embed_dim + fast_dim + recent_dim + video_dim


def _find_stabilizer_output_dim(state_dict):
    output_keys = [
        key
        for key, value in state_dict.items()
        if re.match(r"(module\.)?(stabilizer|forecast_head\.output)\.\d+\.weight$", key)
        and hasattr(value, "ndim")
        and value.ndim == 2
    ]
    if not output_keys:
        return None

    def layer_index(key):
        match = re.search(r"stabilizer\.(\d+)\.weight$", key)
        if match is None:
            match = re.search(r"forecast_head\.output\.(\d+)\.weight$", key)
        return int(match.group(1)) if match else -1

    output_key = max(output_keys, key=layer_index)
    return state_dict[output_key].shape[0]


def _find_slowfast_summary_dim(state_dict):
    for key, value in state_dict.items():
        if re.match(r"(module\.)?forecast_head\.output\.0\.weight$", key):
            return value.shape[1]
    return None


def _check_checkpoint_shape(opt):
    checkpoint = opt["path"]["pretrained"]
    if checkpoint is None:
        return
    if not os.path.exists(checkpoint):
        raise FileNotFoundError("Checkpoint does not exist: {}".format(checkpoint))

    state_dict = torch.load(checkpoint, map_location="cpu")
    if isinstance(state_dict, dict) and "params" in state_dict:
        state_dict = state_dict["params"]

    checkpoint_is_slowfast = any("forecast_head." in key for key in state_dict.keys())
    config_is_slowfast = opt["netG"].get("forecast_head", "legacy") == "slowfast"
    if checkpoint_is_slowfast != config_is_slowfast:
        checkpoint_head = "slowfast" if checkpoint_is_slowfast else "legacy"
        config_head = "slowfast" if config_is_slowfast else "legacy"
        raise ValueError(
            "Checkpoint/config forecasting head mismatch. The checkpoint uses {}, "
            "but the current config builds {}. Re-run with --forecast-head {} or edit "
            "netG.forecast_head in the option file."
            .format(checkpoint_head, config_head, checkpoint_head)
        )

    if config_is_slowfast:
        checkpoint_summary_dim = _find_slowfast_summary_dim(state_dict)
        expected_summary_dim = _slowfast_summary_dim(opt)
        if checkpoint_summary_dim is not None and checkpoint_summary_dim != expected_summary_dim:
            raise ValueError(
                "Checkpoint/config slow-fast head mismatch. The checkpoint summary dimension "
                "is {}, but the current config expects {}. Check slowfast_beta, "
                "slowfast_recent_frames, slowfast_use_recent_delta, and video_model."
                .format(checkpoint_summary_dim, expected_summary_dim)
            )

    checkpoint_dim = _find_stabilizer_output_dim(state_dict)
    expected_dim = _forecast_output_dim(opt)
    if checkpoint_dim is not None and checkpoint_dim != expected_dim:
        per_frame = 7 if opt["datasets"]["train"]["output"] == "aria" else 58
        if not opt["datasets"]["train"]["use_aria"]:
            per_frame = 63
        elif not opt["datasets"]["train"]["use_rot"]:
            per_frame = 3 if opt["datasets"]["train"]["output"] == "aria" else 54
        checkpoint_frames = checkpoint_dim // per_frame if checkpoint_dim % per_frame == 0 else "unknown"
        raise ValueError(
            "Checkpoint/config forecasting horizon mismatch. The checkpoint output head has "
            "{} values, but the current config expects {} values. This usually means "
            "future_frames differs. Checkpoint appears to use future_frames={}; config uses "
            "future_frames={}. Re-run with --future-frames {} or edit the option file."
            .format(
                checkpoint_dim,
                expected_dim,
                checkpoint_frames,
                opt["datasets"]["train"]["future_frames"],
                checkpoint_frames,
            )
        )


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
        "--per-step-csv",
        type=str,
        default=None,
        help="Optional CSV path for per-forecast-step MPJPE, Aria error, and within-horizon velocity error.",
    )
    parser.add_argument(
        "--future-frames",
        type=int,
        default=None,
        help="Optional forecasting horizon override. Applied to train/test config sections.",
    )
    parser.add_argument(
        "--forecast-head",
        type=str,
        choices=["legacy", "slowfast"],
        default=None,
        help="Optional forecasting head override. Must match the checkpoint architecture.",
    )
    parser.add_argument(
        "--slowfast-recent-frames",
        type=int,
        default=None,
        help="Optional number of recent encoded frames concatenated into the slow-fast head.",
    )
    parser.add_argument(
        "--slowfast-use-recent-delta",
        action="store_true",
        help="Concatenate the last encoded frame delta into the slow-fast head.",
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

    if args.future_frames is not None:
        opt["datasets"]["train"]["future_frames"] = args.future_frames
        opt["datasets"]["test"]["future_frames"] = args.future_frames
    if args.forecast_head is not None:
        opt["netG"]["forecast_head"] = args.forecast_head
    if args.slowfast_recent_frames is not None:
        opt["netG"]["slowfast_recent_frames"] = args.slowfast_recent_frames
    if args.slowfast_use_recent_delta:
        opt["netG"]["slowfast_use_recent_delta"] = True

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

    _check_checkpoint_shape(opt)

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
    skipped_nonfinite_velocity = []
    skipped = set(args.skip_indices or [])
    curve_acc = None
    if args.per_step_csv is not None:
        curve_acc = _new_curve_accumulators(opt["datasets"]["test"]["future_frames"])

    with torch.no_grad():
        for index, test_data in enumerate(tqdm(test_loader, desc="Testing")):
            if args.max_samples is not None and len(pos_errors) >= args.max_samples:
                break
            if index in skipped:
                continue

            model.feed_data(test_data, test=True)
            result = model.test_fcast(return_details=curve_acc is not None)

            if curve_acc is not None:
                _add_curve_details(curve_acc, result, opt["datasets"]["test"]["output"])

            if opt["datasets"]["test"]["output"] == "aria":
                pos_error = result["pos_error"] if curve_acc is not None else result[0]
                rot_error = result["rot_error"] if curve_acc is not None else result[1]
                vel_error = result["vel_error"] if curve_acc is not None else result[2]
                rot_errors.append(_as_float(rot_error))
            else:
                pos_error = result["pos_error"] if curve_acc is not None else result[0]
                vel_error = result["vel_error"] if curve_acc is not None else result[1]
                aria_error = result["aria_error"] if curve_acc is not None else result[2]
                aria_errors.append(_as_float(aria_error))

            pos_errors.append(_as_float(pos_error))
            vel_error = _as_float(vel_error)
            if torch.isfinite(torch.tensor(vel_error)):
                vel_errors.append(vel_error)
            else:
                skipped_nonfinite_velocity.append(index)

    if not pos_errors:
        raise RuntimeError("No test samples were evaluated.")
    if not vel_errors:
        raise RuntimeError("No finite velocity errors were evaluated.")

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
    if skipped_nonfinite_velocity:
        skipped_message = (
            "Skipped {:d} non-finite velocity error(s) at test indices: {}"
        ).format(len(skipped_nonfinite_velocity), skipped_nonfinite_velocity)
        logger.warning(skipped_message)
        print(skipped_message)
    if curve_acc is not None:
        _write_curve_csv(args.per_step_csv, curve_acc)
        curve_message = "Wrote per-step horizon curves to {}".format(args.per_step_csv)
        logger.info(curve_message)
        print(curve_message)


if __name__ == "__main__":
    main()
