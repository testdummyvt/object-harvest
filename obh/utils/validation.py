import json
import re
from typing import Any, Dict


def validate_and_clean_prompt_gen_response(response: str) -> Dict[str, Any]:
    """Parse, clean, and validate the LLM response for prompt-gen.

    Args:
        response: Raw LLM response string.

    Returns:
        Validated dict with 'describe' and 'objects'.

    Raises:
        ValueError: If response cannot be validated.
    """
    # Try to extract JSON from response if there's extra text
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
    else:
        json_str = response.strip()

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in response: {response[:200]}...")

    # Validate structure
    if not isinstance(data, dict):
        raise ValueError("Response is not a JSON object")

    if "describe" not in data or "objects" not in data:
        raise ValueError("Missing 'describe' or 'objects' keys")

    if not isinstance(data["describe"], str):
        raise ValueError("'describe' is not a string")

    if not isinstance(data["objects"], list):
        raise ValueError("'objects' is not a list")

    for i, obj in enumerate(data["objects"]):
        if not isinstance(obj, dict):
            raise ValueError(f"objects[{i}] is not a dict")
        if len(obj) != 1:
            raise ValueError(f"objects[{i}] does not have exactly one key-value pair")
        key, value = next(iter(obj.items()))
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError(f"objects[{i}] key or value is not a string")

    return data


def restructure_objects(data: Dict[str, Any]) -> Dict[str, Any]:
    """Restructure the 'objects' key in the prompt-gen response data.

    Converts from list of dicts to dict with 'names' and 'descriptors' lists.

    Args:
        data: The validated response dict.

    Returns:
        The modified data dict.
    """
    objects_list = data["objects"]
    names = [list(obj.keys())[0] for obj in objects_list]
    descriptors = [list(obj.values())[0] for obj in objects_list]
    data["objects"] = {"names": names, "descriptors": descriptors}
    return data