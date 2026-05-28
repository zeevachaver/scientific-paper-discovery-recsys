from datasets import load_dataset

ds = load_dataset(
    "AlgorithmicResearchGroup/s2orc-cs-enriched",
    streaming=True,
    split="train"
)

for i, paper in enumerate(ds):

    print("\n====================")
    print(f"PAPER {i}")
    print("====================")

    print("\nKEYS:")
    print(paper.keys())

    print("\nFULL PAPER:")
    print(paper)

    # -------------------------
    # FIELD BREAKDOWN
    # -------------------------
    print("\nFIELD SUMMARY:")
    for k, v in paper.items():
        if isinstance(v, str):
            preview = v[:200]
            print(f"- {k}: {preview}")
        else:
            print(f"- {k}: {type(v)}")


    # only inspect first 3 papers
    if i >= 2:
        break