"""Image preprocessing utilities for satellite/disaster imagery."""

from PIL import Image
import io

MAX_SIZE = 1024  # Max dimension — prevents GPU OOM on large satellite images


def load_image(path_or_bytes):
    """Load and preprocess an image from a file path or raw bytes.
    
    Automatically:
    - Converts to RGB (handles RGBA, grayscale, etc.)
    - Resizes if any dimension exceeds MAX_SIZE (preserves aspect ratio)
    
    Args:
        path_or_bytes: File path (str) or raw image bytes.
        
    Returns:
        PIL.Image in RGB format, resized if necessary.
    """
    if isinstance(path_or_bytes, bytes):
        image = Image.open(io.BytesIO(path_or_bytes))
    else:
        image = Image.open(path_or_bytes)

    image = image.convert("RGB")

    # Resize large images to prevent GPU OOM
    if max(image.size) > MAX_SIZE:
        image.thumbnail((MAX_SIZE, MAX_SIZE), Image.LANCZOS)

    return image


def get_image_metadata(image: Image.Image) -> dict:
    """Extract basic metadata from a PIL Image."""
    return {
        "width": image.size[0],
        "height": image.size[1],
        "mode": image.mode,
    }
