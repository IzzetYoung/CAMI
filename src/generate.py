from agents import Env, CAMI, Client
import json
from tqdm import tqdm
import os
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, help="OpenAI model to use for the agents")
    parser.add_argument("--retriever_path", type=str, help="The retriever model to use in client simulation.")
    parser.add_argument(
        "--profile_path", type=str, help="Path to the profiles.jsonl file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory to save the generated conversations",
    )
    parser.add_argument(
        "--round", type=int, help="Number of rounds to run the simulation"
    )
    parser.add_argument(
        "--max_turns", type=int, help="Maximum number of turns for each conversation"
    )

    args = parser.parse_args()

    with open(args.profile_path) as f:
        lines = f.readlines()
    for j in range(args.round):
        for i in tqdm(range(len(lines)), desc=f"Round-{j}"):
            sample = json.loads(lines[i])
            if os.path.exists(f"./Output/Sample-{i}-Round-{j}.txt"):
                with open(f"./Output/Sample-{i}-Round-{j}.txt") as f:
                    temp_lines = f.readlines()
                    if (
                        len(temp_lines) > 40
                        or "You are motivated because" in temp_lines[-1]
                        or "You should highlight current state and engagement, express a desire to end the current session"
                        in temp_lines[-1]
                    ):
                        continue
            goal = sample["topic"]
            behavior = sample["Behavior"]
            counselor = CAMI(goal=goal, behavior=behavior, model=args.model)
            reference = ""
            for speaker, utterance in zip(
                sample["speakers"][:50], sample["utterances"][:50]
            ):
                if speaker == "client":
                    reference += f"Client: {utterance}\n"
                else:
                    reference += f"Counselor: {utterance}\n"
            client = Client(
                goal=sample["topic"],
                behavior=sample["Behavior"],
                reference=reference,
                personas=sample["Personas"],
                initial_stage=sample["states"][0],
                final_stage=sample["states"][-1],
                motivation=sample["Motivation"],
                beliefs=sample["Beliefs"],
                plans=sample["Acceptable Plans"],
                receptivity=sum(sample["suggestibilities"])
                / len(sample["suggestibilities"]),
                retriever_path=args.retriever_path,
            )
            env = Env(
                client=client,
                counselor=counselor,
                output_file=f"./Output/Sample-{i}-Round-{j}.txt",
                max_turns=args.max_turns,
            )
            env.interact()
