"""Vision inference pipeline for disaster image analysis using Qwen2-VL."""

from qwen_vl.model_loader import load_model
from qwen_vl.image_processor import load_image
from qwen_vl_utils import process_vision_info

DISASTER_PROMPT = """Analyze this satellite/aerial disaster image carefully. Identify:
1. Flooded areas (approximate bounding polygons as lat/lon coordinates)
2. Damaged roads or infrastructure
3. Collapsed or damaged buildings
4. Severity level (low / medium / high / critical)

Return your analysis as a JSON object with these keys:
- "severity": one of "low", "medium", "high", "critical"
- "findings": a text summary of what you see
- "affected_zones": a list of polygon coordinate arrays, each polygon is a list of [longitude, latitude] pairs

Example format:
{
  "severity": "high",
  "findings": "Significant flooding detected in the southern region...",
  "affected_zones": [
    [[lon1, lat1], [lon2, lat2], [lon3, lat3], [lon1, lat1]]
  ]
}
"""


def analyze_image(image_bytes: bytes, disaster_type: str = "flood") -> str:
    """Run vision inference on a disaster image.
    
    Args:
        image_bytes: Raw image bytes (JPEG/PNG).
        disaster_type: Type of disaster to look for (flood, earthquake, fire, etc.)
        
    Returns:
        Raw text output from the model (should be JSON-formatted).
    """
    model, processor = load_model()
    image = load_image(image_bytes)

    # Build the prompt with disaster context
    prompt = DISASTER_PROMPT.replace("disaster", disaster_type)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Process inputs
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    # Generate
    output_ids = model.generate(**inputs, max_new_tokens=1024)
    output_text = processor.batch_decode(
        output_ids[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )[0]

    return output_text
