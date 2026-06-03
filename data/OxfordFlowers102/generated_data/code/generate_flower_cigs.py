import os
import re
import json
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm
from google import genai
from google.genai import types


def sanitize_filename(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def is_valid_json_file(path: Path) -> bool:
    if not path.exists():
        return False

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            return False

        if "stages" not in data:
            return False

        if not isinstance(data["stages"], list):
            return False

        if len(data["stages"]) != 12:
            return False

        return True

    except Exception:
        return False


def build_prompt(flower_name: str, class_label: int) -> str:
    return f"""
You are a botanical animation planner for the project "SleepingBeauty - Bloom and Fade of a Rose".

Your task is to generate a structured JSON description of the blooming and fading process of a flower species.
The description will be used as motion and appearance guidance for a flower GIF generation system.

The system receives one input image of a flower and generates 24 consecutive animation frames.
The output JSON should describe how the flower visually changes from a closed bud to a fully opened flower and then to a fading or wilted flower.

Flower species: {flower_name}
Class label: {class_label}

Generate a JSON object with the following structure:

{{
  "class_label": {class_label},
  "flower_name": "{flower_name}",
  "total_frames": 24,
  "num_stages": 12,
  "stages": [
    {{
      "stage_id": 1,
      "stage_name": "...",
      "frame_range": "1-2",
      "growth_phase": "...",
      "overall_flower_shape": "...",
      "bud_or_flower_state": "...",
      "petal_description": {{
        "visibility": "...",
        "outer_petals": "...",
        "inner_petals": "...",
        "shape": "...",
        "orientation": "...",
        "opening_degree": "...",
        "curvature": "...",
        "edge_shape": "...",
        "texture": "...",
        "color_transition": "...",
        "motion_direction": "..."
      }},
      "sepal_description": {{
        "visibility": "...",
        "position": "...",
        "orientation": "...",
        "shape": "...",
        "color": "...",
        "motion_direction": "..."
      }},
      "stamen_description": {{
        "visibility": "...",
        "position": "...",
        "shape": "...",
        "color": "...",
        "motion_or_reveal": "..."
      }},
      "pistil_description": {{
        "visibility": "...",
        "position": "...",
        "shape": "...",
        "color": "...",
        "motion_or_reveal": "..."
      }},
      "flower_center_description": "...",
      "symmetry_and_spatial_arrangement": "...",
      "silhouette_change": "...",
      "motion_guidance_for_animation": "...",
      "appearance_guidance_for_generation": "...",
      "negative_constraints": "..."
    }}
  ]
}}

Important requirements:
- Create exactly 12 stages.
- Map the stages to 24 frames:
  stage 1: frames 1-2
  stage 2: frames 3-4
  stage 3: frames 5-6
  stage 4: frames 7-8
  stage 5: frames 9-10
  stage 6: frames 11-12
  stage 7: frames 13-14
  stage 8: frames 15-16
  stage 9: frames 17-18
  stage 10: frames 19-20
  stage 11: frames 21-22
  stage 12: frames 23-24
- The process must cover: closed bud, bud swelling, sepal opening, early petal emergence, partial bloom, half bloom, near full bloom, full bloom, mature bloom, early fading, petal curling or drying, and wilted or petal falling stage.
- Use species-specific visual traits for "{flower_name}".
- Describe petals, sepals, stamens, pistil, flower center, and overall silhouette in detail.
- Make the descriptions useful for image-to-GIF animation.
- Avoid unrealistic motion such as petals stretching like rubber, instant shape changes, or new petals appearing from nowhere.
- Output only valid JSON.
- Do not include markdown.
- Do not include explanations outside the JSON.
""".strip()


def extract_json(text: str) -> dict:
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text)
        text = re.sub(r"```$", "", text)
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


def validate_generated_json(data: dict, class_label: int, flower_name: str) -> None:
    if not isinstance(data, dict):
        raise ValueError("Generated output is not a JSON object")

    if "stages" not in data:
        raise ValueError("Missing key: stages")

    if not isinstance(data["stages"], list):
        raise ValueError("stages must be a list")

    if len(data["stages"]) != 12:
        raise ValueError(f"Expected 12 stages, got {len(data['stages'])}")

    data["class_label"] = int(class_label)
    data["flower_name"] = flower_name
    data["total_frames"] = 24
    data["num_stages"] = 12

    for i, stage in enumerate(data["stages"], start=1):
        if not isinstance(stage, dict):
            raise ValueError(f"Stage {i} is not a JSON object")

        stage["stage_id"] = i
        stage["frame_range"] = f"{2 * i - 1}-{2 * i}"


def call_gemini(
    api_key: str,
    model_name: str,
    prompt: str,
    max_retries: int = 3,
    retry_sleep: float = 2.0,
) -> dict:
    client = genai.Client(api_key=api_key)
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.4,
                    top_p=0.9,
                    response_mime_type="application/json",
                ),
            )

            if not response.text:
                raise ValueError("Empty response from Gemini API")

            return extract_json(response.text)

        except Exception as e:
            last_error = e
            wait_time = retry_sleep * attempt
            time.sleep(wait_time)

    raise RuntimeError(f"Gemini API failed after {max_retries} retries: {last_error}")


def load_class_mapping(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_cols = {"class_label", "name_cat"}
    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        raise ValueError(
            f"CSV file is missing required columns: {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )

    df_unique = (
        df[["class_label", "name_cat"]]
        .drop_duplicates()
        .sort_values("class_label")
        .reset_index(drop=True)
    )

    return df_unique


def process_one_flower(
    row: dict,
    output_dir: Path,
    api_key: str,
    model_name: str,
    force: bool,
    max_retries: int,
) -> dict:
    class_label = int(row["class_label"])
    flower_name = str(row["name_cat"]).strip()

    safe_name = sanitize_filename(flower_name)
    output_path = output_dir / f"{class_label:03d}_{safe_name}.json"
    error_path = output_dir / f"{class_label:03d}_{safe_name}_ERROR.txt"

    if not force and is_valid_json_file(output_path):
        return {
            "status": "skipped",
            "class_label": class_label,
            "flower_name": flower_name,
            "path": str(output_path),
        }

    prompt = build_prompt(
        flower_name=flower_name,
        class_label=class_label,
    )

    try:
        flower_json = call_gemini(
            api_key=api_key,
            model_name=model_name,
            prompt=prompt,
            max_retries=max_retries,
        )

        validate_generated_json(
            data=flower_json,
            class_label=class_label,
            flower_name=flower_name,
        )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(flower_json, f, ensure_ascii=False, indent=2)

        if error_path.exists():
            error_path.unlink()

        return {
            "status": "generated",
            "class_label": class_label,
            "flower_name": flower_name,
            "path": str(output_path),
        }

    except Exception as e:
        with open(error_path, "w", encoding="utf-8") as f:
            f.write(str(e))

        return {
            "status": "failed",
            "class_label": class_label,
            "flower_name": flower_name,
            "error": str(e),
            "path": str(error_path),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Generate flower blooming JSON files using Gemini API with multi-worker support."
    )

    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to class_mapping.csv",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Folder to save generated JSON files",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gemini-3.1-flash-lite",
        help="Gemini model name",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for testing, e.g. --limit 5",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel API workers",
    )

    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Max retries for each flower",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate all files even if valid JSON already exists",
    )

    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Missing GEMINI_API_KEY. Set it first:\n"
            'export GEMINI_API_KEY="YOUR_API_KEY"'
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_flowers = load_class_mapping(args.csv_path)

    if args.limit is not None:
        df_flowers = df_flowers.head(args.limit)

    rows = df_flowers.to_dict(orient="records")

    print(f"Loaded flower classes: {len(rows)}")
    print(f"Output folder: {output_dir}")
    print(f"Model: {args.model}")
    print(f"Num workers: {args.num_workers}")
    print(f"Force regenerate: {args.force}")

    summary = {
        "generated": 0,
        "skipped": 0,
        "failed": 0,
    }

    failed_items = []

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [
            executor.submit(
                process_one_flower,
                row=row,
                output_dir=output_dir,
                api_key=api_key,
                model_name=args.model,
                force=args.force,
                max_retries=args.max_retries,
            )
            for row in rows
        ]

        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            status = result["status"]

            summary[status] += 1

            if status == "failed":
                failed_items.append(result)
                print(
                    f"[FAILED] {result['class_label']:03d} "
                    f"{result['flower_name']} -> {result['error']}"
                )

    print("\nDone.")
    print(f"Generated: {summary['generated']}")
    print(f"Skipped:   {summary['skipped']}")
    print(f"Failed:    {summary['failed']}")

    if failed_items:
        failed_log_path = output_dir / "failed_summary.json"
        with open(failed_log_path, "w", encoding="utf-8") as f:
            json.dump(failed_items, f, ensure_ascii=False, indent=2)

        print(f"Failed summary saved to: {failed_log_path}")


if __name__ == "__main__":
    main()