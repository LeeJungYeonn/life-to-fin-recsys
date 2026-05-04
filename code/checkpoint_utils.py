import hashlib
import json
import os
import pickle
import random
from pathlib import Path

import torch

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

from models import SourceEncoder, TargetEncoder


def _torch_load_compat(path, map_location="cpu", *, weights_only=True):
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:  # pragma: no cover
        return torch.load(path, map_location=map_location)
    except pickle.UnpicklingError:
        if not weights_only:
            raise
        return torch.load(path, map_location=map_location, weights_only=False)


def set_reproducible_mode(seed, deterministic=True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    if np is not None:
        np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False

    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False

    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(deterministic, warn_only=not deterministic)
        except TypeError:  # pragma: no cover
            torch.use_deterministic_algorithms(deterministic)


def make_torch_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def capture_rng_state(dataloader_generator=None):
    state = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if np is not None:
        state["numpy"] = np.random.get_state()
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if dataloader_generator is not None:
        state["dataloader_generator"] = dataloader_generator.get_state()
    return state


def restore_rng_state(rng_state, dataloader_generator=None):
    if not rng_state:
        return

    python_state = rng_state.get("python")
    if python_state is not None:
        random.setstate(python_state)

    numpy_state = rng_state.get("numpy")
    if numpy_state is not None and np is not None:
        np.random.set_state(numpy_state)

    torch_state = rng_state.get("torch")
    if torch_state is not None:
        torch.set_rng_state(torch_state)

    cuda_state = rng_state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)

    generator_state = rng_state.get("dataloader_generator")
    if generator_state is not None and dataloader_generator is not None:
        dataloader_generator.set_state(generator_state)


def _tensor_sha256(tensor):
    if tensor is None:
        return None
    tensor = tensor.detach().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def _tensor_shape(tensor):
    if tensor is None:
        return None
    return list(tensor.shape)


def _tensor_max_per_col(tensor):
    if tensor is None:
        return None
    if tensor.ndim != 2:
        return None
    return torch.amax(tensor.detach().cpu(), dim=0).tolist()


def _validate_tensor_identity(name, tensor, saved_shape, saved_hash, errors):
    current_shape = _tensor_shape(tensor)
    if current_shape != saved_shape:
        errors.append(
            f"Current {name} shape does not match the checkpoint. "
            f"(current={current_shape}, saved={saved_shape})"
        )
        return

    current_hash = _tensor_sha256(tensor)
    if saved_hash is not None and current_hash != saved_hash:
        errors.append(f"Current {name} content hash does not match the checkpoint.")


def build_preprocess_info(
    categorical_cols,
    categorical_cardinalities,
    x_cat_tensor=None,
    x_ratio_tensor=None,
    labels_tensor=None,
    split=None,
):
    return {
        "version": 1,
        "split": split,
        "categorical_cols": list(categorical_cols),
        "categorical_cardinalities": [int(v) for v in categorical_cardinalities],
        "num_source_features": len(categorical_cols),
        "x_cat_shape": _tensor_shape(x_cat_tensor),
        "x_cat_hash": _tensor_sha256(x_cat_tensor),
        "x_cat_max_per_col": _tensor_max_per_col(x_cat_tensor),
        "x_ratio_shape": _tensor_shape(x_ratio_tensor),
        "x_ratio_hash": _tensor_sha256(x_ratio_tensor),
        "labels_shape": _tensor_shape(labels_tensor),
        "labels_hash": _tensor_sha256(labels_tensor),
    }


def save_dual_encoder_checkpoint(
    checkpoint_dir,
    source_encoder,
    target_encoder,
    preprocess_info,
    model_config,
    best_params=None,
    best_loss=None,
    optimizer=None,
    epoch=None,
    rng_state=None,
    seed=None,
    deterministic=None,
    prefix="best",
):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    source_path = checkpoint_dir / f"{prefix}_source_encoder.pth"
    target_path = checkpoint_dir / f"{prefix}_target_encoder.pth"
    meta_path = checkpoint_dir / f"{prefix}_checkpoint_meta.json"
    training_state_path = checkpoint_dir / f"{prefix}_training_state.pth"

    torch.save(source_encoder.state_dict(), source_path)
    torch.save(target_encoder.state_dict(), target_path)

    payload = {
        "source_checkpoint": source_path.name,
        "target_checkpoint": target_path.name,
        "model_config": dict(model_config),
        "preprocess_info": dict(preprocess_info),
        "best_params": best_params,
        "best_loss": best_loss,
    }

    training_state = {
        "epoch": epoch,
        "seed": seed,
        "deterministic": deterministic,
        "torch_version": torch.__version__,
    }
    if optimizer is not None:
        training_state["optimizer_state_dict"] = optimizer.state_dict()
    if rng_state is not None:
        training_state["rng_state"] = rng_state

    if any(value is not None for key, value in training_state.items() if key != "torch_version"):
        torch.save(training_state, training_state_path)
        payload["training_state_checkpoint"] = training_state_path.name

    meta_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def load_checkpoint_meta(checkpoint_dir, prefix="best"):
    meta_path = Path(checkpoint_dir) / f"{prefix}_checkpoint_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Checkpoint metadata not found: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def validate_preprocess_info(
    checkpoint_meta,
    current_categorical_cols=None,
    current_categorical_cardinalities=None,
    current_x_cat_tensor=None,
    current_x_ratio_tensor=None,
    current_labels_tensor=None,
    current_split=None,
):
    preprocess_info = checkpoint_meta["preprocess_info"]
    model_config = checkpoint_meta["model_config"]

    errors = []
    warnings = []

    saved_cols = preprocess_info["categorical_cols"]
    saved_cardinalities = preprocess_info["categorical_cardinalities"]
    saved_split = preprocess_info.get("split")

    if current_categorical_cols is not None:
        if list(current_categorical_cols) != saved_cols:
            errors.append("Categorical column order does not match the training checkpoint.")

    if current_categorical_cardinalities is not None:
        current_categorical_cardinalities = [int(v) for v in current_categorical_cardinalities]
        if len(current_categorical_cardinalities) != len(saved_cardinalities):
            errors.append("Current categorical cardinality length does not match the checkpoint.")
        else:
            larger_cols = [
                saved_cols[idx]
                for idx, (current_v, saved_v) in enumerate(
                    zip(current_categorical_cardinalities, saved_cardinalities)
                )
                if current_v > saved_v
            ]
            if larger_cols:
                errors.append(
                    "Current categorical cardinalities exceed the training checkpoint for: "
                    + ", ".join(larger_cols)
                )
            elif current_categorical_cardinalities != saved_cardinalities:
                warnings.append(
                    "Current categorical cardinalities differ from the training checkpoint. "
                    "This can be normal for eval splits with missing categories."
                )

    if current_x_cat_tensor is not None:
        if current_x_cat_tensor.ndim != 2:
            errors.append("Current x_cat_tensor must be 2D.")
        elif current_x_cat_tensor.shape[1] != len(saved_cols):
            errors.append("Current x_cat_tensor feature count does not match the checkpoint.")
        else:
            current_max = torch.amax(current_x_cat_tensor.detach().cpu(), dim=0).tolist()
            exceeded_cols = [
                saved_cols[idx]
                for idx, (max_v, saved_v) in enumerate(zip(current_max, saved_cardinalities))
                if max_v >= saved_v
            ]
            if exceeded_cols:
                errors.append(
                    "Current x_cat_tensor contains category indices outside the training checkpoint range for: "
                    + ", ".join(exceeded_cols)
                )

    if current_x_ratio_tensor is not None:
        if current_x_ratio_tensor.ndim != 2:
            errors.append("Current x_ratio_tensor must be 2D.")
        elif current_x_ratio_tensor.shape[1] != int(model_config["target_input_dim"]):
            errors.append("Current x_ratio_tensor feature count does not match the checkpoint.")

    if current_labels_tensor is not None and current_x_cat_tensor is not None:
        if len(current_labels_tensor) != len(current_x_cat_tensor):
            errors.append("Current labels_tensor length does not match x_cat_tensor length.")

    exact_check_requested = any(
        tensor is not None
        for tensor in (current_x_cat_tensor, current_x_ratio_tensor, current_labels_tensor)
    )
    if exact_check_requested:
        if current_split is None:
            warnings.append("Exact tensor hash validation was skipped because current_split was not provided.")
        elif current_split != saved_split:
            warnings.append(
                f"Current split '{current_split}' differs from checkpoint split '{saved_split}'. "
                "Exact tensor hash validation was skipped."
            )
        else:
            if current_x_cat_tensor is not None:
                _validate_tensor_identity(
                    "x_cat_tensor",
                    current_x_cat_tensor,
                    preprocess_info["x_cat_shape"],
                    preprocess_info["x_cat_hash"],
                    errors,
                )
            if current_x_ratio_tensor is not None:
                _validate_tensor_identity(
                    "x_ratio_tensor",
                    current_x_ratio_tensor,
                    preprocess_info["x_ratio_shape"],
                    preprocess_info["x_ratio_hash"],
                    errors,
                )
            if current_labels_tensor is not None:
                _validate_tensor_identity(
                    "labels_tensor",
                    current_labels_tensor,
                    preprocess_info["labels_shape"],
                    preprocess_info["labels_hash"],
                    errors,
                )

    return {
        "errors": errors,
        "warnings": warnings,
        "saved_preprocess_info": preprocess_info,
    }


def load_dual_encoder_checkpoint(
    checkpoint_dir,
    device,
    current_categorical_cols=None,
    current_categorical_cardinalities=None,
    current_x_cat_tensor=None,
    current_x_ratio_tensor=None,
    current_labels_tensor=None,
    current_split=None,
    strict=True,
    prefix="best",
    optimizer=None,
    restore_training_state=False,
    dataloader_generator=None,
):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_meta = load_checkpoint_meta(checkpoint_dir, prefix=prefix)
    validation = validate_preprocess_info(
        checkpoint_meta,
        current_categorical_cols=current_categorical_cols,
        current_categorical_cardinalities=current_categorical_cardinalities,
        current_x_cat_tensor=current_x_cat_tensor,
        current_x_ratio_tensor=current_x_ratio_tensor,
        current_labels_tensor=current_labels_tensor,
        current_split=current_split,
    )

    if strict and validation["errors"]:
        raise ValueError("\n".join(validation["errors"]))

    model_config = checkpoint_meta["model_config"]
    preprocess_info = checkpoint_meta["preprocess_info"]

    source_encoder = SourceEncoder(
        preprocess_info["categorical_cardinalities"],
        embed_dim=int(model_config["embed_dim"]),
        output_dim=int(model_config["output_dim"]),
        projection_dim=int(model_config.get("projection_dim", 64)),
        num_risk_levels=int(model_config.get("num_risk_levels", 5)),
        ratio_dim=int(model_config.get("ratio_dim", model_config["target_input_dim"])),
    ).to(device)
    target_encoder = TargetEncoder(
        input_dim=int(model_config["target_input_dim"]),
        output_dim=int(model_config["output_dim"]),
        projection_dim=int(model_config.get("projection_dim", 64)),
        num_risk_levels=int(model_config.get("num_risk_levels", 5)),
        ratio_dim=int(model_config.get("ratio_dim", model_config["target_input_dim"])),
    ).to(device)

    source_state = _torch_load_compat(
        checkpoint_dir / checkpoint_meta["source_checkpoint"],
        map_location=device,
        weights_only=True,
    )
    target_state = _torch_load_compat(
        checkpoint_dir / checkpoint_meta["target_checkpoint"],
        map_location=device,
        weights_only=True,
    )

    source_encoder.load_state_dict(source_state)
    target_encoder.load_state_dict(target_state)

    training_state_path = checkpoint_meta.get("training_state_checkpoint")
    training_state = None
    if training_state_path is not None:
        training_state = _torch_load_compat(
            checkpoint_dir / training_state_path,
            map_location="cpu",
            weights_only=False,
        )
        if optimizer is not None and training_state.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(training_state["optimizer_state_dict"])
        if restore_training_state:
            restore_rng_state(
                training_state.get("rng_state"),
                dataloader_generator=dataloader_generator,
            )

    checkpoint_meta["training_state"] = training_state

    source_encoder.eval()
    target_encoder.eval()

    return source_encoder, target_encoder, checkpoint_meta, validation
