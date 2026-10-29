import argparse
import json
import os

from tqdm import tqdm

from agentless.util.utils import load_jsonl


RRF_K = 10


def fuse_ranked_lists(model_files, retrieval_files, top_n, k=RRF_K):
    scores = {}
    first_seen_order = {}

    for ranked_files in (model_files[:top_n], retrieval_files[:top_n]):
        seen_in_ranking = set()
        for rank, file_path in enumerate(ranked_files, start=1):
            # A file has one rank per source. Keep its first rank if an input
            # contains duplicates, and let duplicates still consume top_n slots.
            if file_path in seen_in_ranking:
                continue
            seen_in_ranking.add(file_path)

            scores[file_path] = scores.get(file_path, 0.0) + 1 / (k + rank)
            if file_path not in first_seen_order:
                first_seen_order[file_path] = len(first_seen_order)

    # Break equal-score ties by first encounter: LLM order first, then retrieval
    # order, matching the source iteration order used by the baseline combiner.
    ranked_files = sorted(
        scores,
        key=lambda file_path: (-scores[file_path], first_seen_order[file_path]),
    )
    return ranked_files, scores


def combine_file_level(args):
    embed_used_locs = load_jsonl(args.retrieval_loc_file)
    model_used_locs = load_jsonl(args.model_loc_file)

    with open(f"{args.output_folder}/embed_used_locs.jsonl", "w") as file:
        for loc in embed_used_locs:
            file.write(json.dumps(loc) + "\n")

    with open(f"{args.output_folder}/model_used_locs.jsonl", "w") as file:
        for loc in model_used_locs:
            file.write(json.dumps(loc) + "\n")

    with open(args.output_file, "w") as output_file, open(
        args.debug_output_file, "w"
    ) as debug_file:
        for pred in tqdm(model_used_locs, colour="MAGENTA"):
            instance_id = pred["instance_id"]
            model_files = pred["found_files"]
            retrieval_files = [
                loc for loc in embed_used_locs if loc["instance_id"] == instance_id
            ][0]["found_files"]

            ranked_files, scores = fuse_ranked_lists(
                model_files, retrieval_files, args.top_n
            )
            output_file.write(
                json.dumps(
                    {
                        "instance_id": instance_id,
                        "found_files": ranked_files,
                        "additional_artifact_loc_file": {},
                        "file_traj": {},
                    }
                )
                + "\n"
            )
            debug_file.write(
                json.dumps(
                    {
                        "instance_id": instance_id,
                        "rrf_scores": {
                            file_path: scores[file_path] for file_path in ranked_files
                        },
                    }
                )
                + "\n"
            )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="combined_locs.jsonl")
    parser.add_argument("--retrieval_loc_file", type=str, required=True)
    parser.add_argument("--model_loc_file", type=str, required=True)
    parser.add_argument("--top_n", type=int, required=True)
    parser.add_argument(
        "--debug_output_file",
        type=str,
        default="combined_locs_rrf_debug.jsonl",
        help="Separate JSONL file containing per-file RRF scores.",
    )

    args = parser.parse_args()
    args.output_file = os.path.join(args.output_folder, args.output_file)
    args.debug_output_file = os.path.join(
        args.output_folder, args.debug_output_file
    )
    assert not os.path.exists(args.output_file), "Output file already exists"
    assert not os.path.exists(args.debug_output_file), "Debug output file already exists"
    assert args.output_file != args.debug_output_file, "Output files must be different"

    os.makedirs(args.output_folder, exist_ok=True)

    with open(os.path.join(args.output_folder, "args.json"), "w") as file:
        json.dump(vars(args), file, indent=4)

    combine_file_level(args)


if __name__ == "__main__":
    main()
