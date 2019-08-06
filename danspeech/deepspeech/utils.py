def get_default_audio_config():
    return {
        "normalize": True,
        "sampling_rate": 16000,
        "window": "hamming",
        "window_stride": 0.01,
        "window_size": 0.02
    }
