"""Print and enforce the configured Paddle and ONNX Runtime backend."""

from __future__ import annotations

import json


def main() -> None:
    import onnxruntime as ort
    import paddle

    from app.config import Settings
    from app.models import Models

    settings = Settings.from_env()
    if settings.runtime.target == "gpu":
        paddle.set_device(f"gpu:{settings.runtime.gpu_id}")
    models = Models(settings)
    runner = models.profile_batch_runner()
    # This raises if CUDA is unavailable or either initialized localizer chose CPU.
    runtime = models.readiness()
    report = {
        "runtime": runtime,
        "paddle": {
            "version": paddle.__version__,
            "compiled_with_cuda": paddle.device.is_compiled_with_cuda(),
            "device_count": paddle.device.cuda.device_count(),
            "selected_device": paddle.device.get_device(),
        },
        "onnxruntime": {
            "version": ort.__version__,
            "available_providers": ort.get_available_providers(),
            "localization_active_providers": {
                name: localizer.providers for name, localizer in runner.localizers.items()
            },
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
