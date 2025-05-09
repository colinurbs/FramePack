bucket_options = {
    128: [
        (96, 160),
        (112, 144),
        (128, 128),
        (144, 112),
        (160, 96),
    ],
    256: [
        (192, 320),
        (224, 288),
        (256, 256),
        (288, 224),
        (320, 192),
    ],
    384: [
        (256, 512),
        (320, 448),
        (384, 384),
        (448, 320),
        (512, 256),
    ],
    512: [
        (352, 704),
        (384, 640),
        (448, 576),
        (512, 512),
        (576, 448),
        (640, 384),
        (704, 352),
    ],
    640: [
        (416, 960),
        (448, 864),
        (480, 832),
        (512, 768),
        (544, 704),
        (576, 672),
        (608, 640),
        (640, 640),
        (640, 608),
        (672, 576),
        (704, 544),
        (768, 512),
        (832, 480),
        (864, 448),
        (960, 416),
    ],
    768: [
        (512, 1024),
        (576, 896),
        (640, 832),
        (704, 768),
        (768, 768),
        (768, 704),
        (832, 640),
        (896, 576),
        (1024, 512),
    ],
}


def find_nearest_bucket(h, w, resolution=640):
    # Use the provided resolution or find the closest available bucket size
    # print(f"find_nearest_bucket called with h={h}, w={w}, resolution={resolution}")
    
    if resolution not in bucket_options:
        # Find the closest available resolution
        available_resolutions = list(bucket_options.keys())
        closest_resolution = min(available_resolutions, key=lambda x: abs(x - resolution))
        # print(f"Resolution {resolution} not found in bucket options, using closest available: {closest_resolution}")
        resolution = closest_resolution
    # else:
        # print(f"Resolution {resolution} found in bucket options")
    
    # Calculate the aspect ratio of the input image
    input_aspect_ratio = w / h if h > 0 else 1.0
    # print(f"Input aspect ratio: {input_aspect_ratio:.4f}")
    
    min_diff = float('inf')
    best_bucket = None
    
    # Find the bucket size with the closest aspect ratio to the input image
    for (bucket_h, bucket_w) in bucket_options[resolution]:
        bucket_aspect_ratio = bucket_w / bucket_h if bucket_h > 0 else 1.0
        # Calculate the difference in aspect ratios
        diff = abs(bucket_aspect_ratio - input_aspect_ratio)
        if diff < min_diff:
            min_diff = diff
            best_bucket = (bucket_h, bucket_w)
        # print(f"  Checking bucket ({bucket_h}, {bucket_w}), aspect ratio={bucket_aspect_ratio:.4f}, diff={diff:.4f}, current best={best_bucket}")
    
    # print(f"Using resolution {resolution}, selected bucket: {best_bucket}")
    return best_bucket
