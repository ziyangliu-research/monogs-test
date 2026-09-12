import numpy as np
import torch
from torch import nn

import slam as monogs_slam
import run_tartanair_holdout_safe as safe_runner
from gaussian_splatting.scene.gaussian_model import GaussianModel


_original_extend_from_pcd_seq = GaussianModel.extend_from_pcd_seq


def _brief_tensor(tensor):
    info = {
        "shape": tuple(tensor.shape),
        "numel": int(tensor.numel()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "contiguous": bool(tensor.is_contiguous()),
    }
    if tensor.numel() > 0 and (tensor.is_floating_point() or tensor.is_complex()):
        finite = torch.isfinite(tensor)
        info["finite"] = bool(finite.all().item())
        if tensor.is_floating_point() and info["finite"]:
            info["min"] = float(tensor.min().item())
            info["max"] = float(tensor.max().item())
    return info


def _diag_extend_from_pcd_seq(
    self, cam_info, kf_id=-1, init=False, scale=2.0, depthmap=None
):
    self._diag_current_kf_id = int(kf_id)
    depth_msg = "depthmap=None"
    if depthmap is not None:
        depth_np = np.asarray(depthmap)
        valid = np.isfinite(depth_np) & (depth_np > 0)
        if valid.any():
            vals = depth_np[valid]
            depth_msg = (
                f"depth_valid={int(valid.sum())}/{int(valid.size)} "
                f"({100.0 * valid.mean():.3f}%) "
                f"min/med/max={float(vals.min()):.6f}/"
                f"{float(np.median(vals)):.6f}/{float(vals.max()):.6f}"
            )
        else:
            depth_msg = f"depth_valid=0/{int(valid.size)}"

    print(
        f"[MONOGS-DIAG] extend_from_pcd_seq ENTER "
        f"kf_id={kf_id} cam_uid={getattr(cam_info, 'uid', None)} "
        f"map_points={int(self.get_xyz.shape[0])} init={init} {depth_msg}",
        flush=True,
    )
    torch.cuda.synchronize()
    out = _original_extend_from_pcd_seq(
        self, cam_info, kf_id=kf_id, init=init, scale=scale, depthmap=depthmap
    )
    torch.cuda.synchronize()
    print(
        f"[MONOGS-DIAG] extend_from_pcd_seq EXIT "
        f"kf_id={kf_id} map_points={int(self.get_xyz.shape[0])}",
        flush=True,
    )
    return out


def _diag_cat_tensors_to_optimizer(self, tensors_dict):
    """Released MonoGS logic plus logging/synchronization only."""
    optimizable_tensors = {}
    kf_id = getattr(self, "_diag_current_kf_id", None)

    for group in self.optimizer.param_groups:
        assert len(group["params"]) == 1
        name = group["name"]
        param = group["params"][0]
        extension_tensor = tensors_dict[name]
        stored_state = self.optimizer.state.get(param, None)

        print(
            f"[MONOGS-DIAG] CAT BEGIN kf_id={kf_id} group={name} "
            f"param={_brief_tensor(param)} ext={_brief_tensor(extension_tensor)} "
            f"state={'yes' if stored_state is not None else 'no'}",
            flush=True,
        )
        torch.cuda.synchronize()

        if stored_state is not None:
            print(
                f"[MONOGS-DIAG] STATE kf_id={kf_id} group={name} "
                f"exp_avg={_brief_tensor(stored_state['exp_avg'])} "
                f"exp_avg_sq={_brief_tensor(stored_state['exp_avg_sq'])}",
                flush=True,
            )

            zeros = torch.zeros_like(extension_tensor)
            torch.cuda.synchronize()
            print(
                f"[MONOGS-DIAG] zeros_like OK kf_id={kf_id} group={name} "
                f"zeros={_brief_tensor(zeros)}",
                flush=True,
            )

            try:
                new_exp_avg = torch.cat((stored_state["exp_avg"], zeros), dim=0)
                torch.cuda.synchronize()
                print(
                    f"[MONOGS-DIAG] exp_avg CAT OK kf_id={kf_id} group={name} "
                    f"out={_brief_tensor(new_exp_avg)}",
                    flush=True,
                )
            except Exception:
                print(
                    f"[MONOGS-DIAG] exp_avg CAT FAILED kf_id={kf_id} group={name}",
                    flush=True,
                )
                raise

            zeros_sq = torch.zeros_like(extension_tensor)
            torch.cuda.synchronize()
            try:
                new_exp_avg_sq = torch.cat(
                    (stored_state["exp_avg_sq"], zeros_sq), dim=0
                )
                torch.cuda.synchronize()
                print(
                    f"[MONOGS-DIAG] exp_avg_sq CAT OK kf_id={kf_id} group={name} "
                    f"out={_brief_tensor(new_exp_avg_sq)}",
                    flush=True,
                )
            except Exception:
                print(
                    f"[MONOGS-DIAG] exp_avg_sq CAT FAILED kf_id={kf_id} group={name}",
                    flush=True,
                )
                raise

            try:
                new_param = torch.cat((param, extension_tensor), dim=0)
                torch.cuda.synchronize()
                print(
                    f"[MONOGS-DIAG] param CAT OK kf_id={kf_id} group={name} "
                    f"out={_brief_tensor(new_param)}",
                    flush=True,
                )
            except Exception:
                print(
                    f"[MONOGS-DIAG] param CAT FAILED kf_id={kf_id} group={name}",
                    flush=True,
                )
                raise

            stored_state["exp_avg"] = new_exp_avg
            stored_state["exp_avg_sq"] = new_exp_avg_sq
            del self.optimizer.state[param]
            group["params"][0] = nn.Parameter(new_param.requires_grad_(True))
            self.optimizer.state[group["params"][0]] = stored_state
            optimizable_tensors[name] = group["params"][0]
        else:
            try:
                new_param = torch.cat((param, extension_tensor), dim=0)
                torch.cuda.synchronize()
                print(
                    f"[MONOGS-DIAG] param CAT(no-state) OK "
                    f"kf_id={kf_id} group={name} out={_brief_tensor(new_param)}",
                    flush=True,
                )
            except Exception:
                print(
                    f"[MONOGS-DIAG] param CAT(no-state) FAILED "
                    f"kf_id={kf_id} group={name}",
                    flush=True,
                )
                raise

            group["params"][0] = nn.Parameter(new_param.requires_grad_(True))
            optimizable_tensors[name] = group["params"][0]

        print(
            f"[MONOGS-DIAG] CAT END kf_id={kf_id} group={name}", flush=True
        )

    return optimizable_tensors


# Top-level monkey patches are intentional: multiprocessing spawn imports this
# module in child processes too, so the backend receives the same diagnostics.
GaussianModel.extend_from_pcd_seq = _diag_extend_from_pcd_seq
GaussianModel.cat_tensors_to_optimizer = _diag_cat_tensors_to_optimizer


if __name__ == "__main__":
    monogs_slam.wandb.log = lambda *args, **kwargs: None
    safe_runner.main()
