from pathlib import Path
import numpy as np
from PIL import Image
import pandas as pd 
import matplotlib.pyplot as plt
from exp_config import POSSIBLE_DATASETS
import pickle

exp_path = Path("./experiments")
output_dir = Path("./experiments/extra/bad_images")
output_dir.mkdir(exist_ok=True)

n = 5
dataset = pickle.load(open(exp_path / "imagenet" / "data" / "test_images.pkl", "rb"))

for model in (exp_path / "imagenet").iterdir():  
    if model.stem == "data" or model.stem == "paper_imgs":
        continue

    model_res = {}
    methods = ["exp_ranem", "exp_intgrad", "exp_rise", "exp_saliency", "exp_gradcam","exp_nemt", "exp_iba","exp_gradshap", "exp_smoothpixelmask"]
    for method in methods :
        method = model / method
        res_path = method / "faithfullness.npy"
        if not res_path.exists():
            print(f"Skipping {res_path} as it does not exist")
            continue

        data = np.load(res_path)[:,1]
        model_res[method.stem] = data

    df = pd.DataFrame(model_res).astype(np.float32)
    df["mean_vs_ranem"] = df.drop(columns=["exp_ranem"]).mean(axis=1) - df["exp_ranem"] 
    df = df.sort_values("mean_vs_ranem", ascending=False).head(n)
    img_ids = list(df.index)
    
    for img_id in img_ids:
        base_img = dataset[img_id][0].resize((224, 224))
        paths = [
            model / f"exp_ranem/attr/{img_id}.png",
            model / f"exp_intgrad/attr/{img_id}.png",
            model / f"exp_rise/attr/{img_id}.png"
        ]
        
        titles = ["exp_ranem", "exp_intgrad", "exp_rise"]
        pretty_titles = ["Ra-NEM", "Integrated Gradients", "RISE"]
        
        images = []
        for path in paths:
            if not path.exists():
                print(f"Missing image: {path}")
                images.append(None)
            else:
                images.append(Image.open(path))
        
        # Create the figure and axes
        fig, axes = plt.subplots(1, len(images), figsize=(15, 5))

        for ax, img, title in zip(axes, images, titles):
            if img is not None:
                ax.imshow(base_img)
                if title == "ranem":
                    img = np.array(img).astype(np.float32) ** 4
                ax.imshow(img, alpha=0.5, cmap="hot")
                
                    

                # Get and format the score for this method
                if title in df.columns:
                    score = df.loc[ img_id, title]
                    ax.set_title(f"{pretty_titles[titles.index(title)]}\nscore: {score:.3f}")
                else:
                    ax.set_title(title)
            ax.axis("off")

        plt.tight_layout()

        # Save the figure
        output_path = output_dir / f"{model.stem}_{img_id}.png"
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved {output_path}")
        plt.close(fig)
