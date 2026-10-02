import torch
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(os.path.dirname(BASE_DIR), "models")

files = [
    "spatial_resnet50_v2.pth",
    "frequency_resnet50_v2.pth",
    "hybrid_resnet50_fft_v2.pth",
]

print("=" * 70)
print("CHECKING V2 MODEL CHECKPOINTS")
print("=" * 70)

for filename in files:

    path = os.path.join(MODEL_DIR, filename)

    print("\n" + "=" * 70)
    print(filename)
    print("=" * 70)

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    print("Checkpoint type:")
    print(type(checkpoint))

    if isinstance(checkpoint, dict):

        print("\nTop-level keys:")
        for key in checkpoint.keys():
            print("  ", key)

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint["model_state_dict"]

            print("\nUsing: model_state_dict")

        elif "state_dict" in checkpoint:

            state_dict = checkpoint["state_dict"]

            print("\nUsing: state_dict")

        else:

            state_dict = checkpoint

            print("\nUsing checkpoint directly")

        print("\nNumber of state-dict keys:")
        print(len(state_dict))

        print("\nFirst 20 keys:")

        for i, key in enumerate(state_dict.keys()):

            if i >= 20:
                break

            print(" ", key)

    else:

        print("Checkpoint is not a dictionary.")

print("\n" + "=" * 70)
print("CHECK COMPLETE")
print("=" * 70)