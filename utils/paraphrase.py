
import sys
import argparse
from pathlib import Path
from openai import OpenAI

client = OpenAI()

def paraphrase(string, num=1):
    messages = [
        {"role": "system", "content": "Please paraphrase the following instructions while strictly adhering to their meaning. Any words surrounded by {} should also appear in your result with a similar context."},
        {"role": "user", "content": string}
    ]
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages,
        temperature=0.7,
        n=num,
    )
    return [choice.message.content for choice in response.choices]

if __name__ == "__main__":
    """
    Example usage:
    python paraphrase.py prompts_constructive/initial_system.txt -n 3
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="Path to file containing content to paraphrase")
    parser.add_argument("-n", "--num", type=int, default=1, help="Number of paraphrases to generate")
    args = parser.parse_args()
    filename, num = Path(args.filename), args.num

    with open(filename, "r") as f:
        responses = paraphrase(f.read(), num)
    for i, response in enumerate(responses):
        with open(filename.parent / Path(str(filename.stem) + f"-{i}" + str(filename.suffix)), "w") as f:
            f.write(response)