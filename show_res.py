import os
import numpy as np
from pathlib import Path
import pandas as pd
from scipy.stats import wilcoxon
import argparse
import sys

if __name__ != "__main__":
    sys.exit()
parser = argparse.ArgumentParser(description="return full res or main res")
parser.add_argument("--verbose", action="store_true", help="Enable verbose mode")
parser.add_argument("--significance", action="store_true", help="Enable significance testing")
args = parser.parse_args()
only_main = not args.verbose
show_significance = args.significance
main_methods = [
    "gradcam",
    "gradshap",
    "gradshap_forgrad",
    "intgrad",
    "intgrad_forgrad",
    "nemt",
    "ranem",
    "rise",
    "smoothpixelmask",
    "saliency",
    "saliency_forgrad",
    "iba",
]
main_methods = ["exp_" + m for m in main_methods]

args = parser.parse_args()
exp_path = Path("./experiments")
model_training_time_path = Path("./attrs/nem_utils/logs")
exp_extra = Path("./experiments/extra")
print("Experiment:")
for file in model_training_time_path.iterdir():
    print(file.stem)

res = []
print("Robustness res")
for model in (Path(exp_path) / "imagenet").iterdir():  # Adjust the path as needed
    if model.stem == "data":
        continue
    # Loop through each method within the model
    for method in model.iterdir():
        res_path = method / "robustness.npy"
        if not res_path.exists():
            #   print(f"Skipping {res_path} as it does not exist")
            continue
        data = np.load(res_path)
        res.append((model.stem, method.stem, np.nanmean(data)))
res = pd.DataFrame(res, columns=["model", "method", "data"])
res = res[res.method.isin(main_methods)] if only_main else res
average_res = res.groupby("method").data.agg(["mean", "count"]).reset_index()
average_res
print(average_res.to_markdown(index=False))

res = []
print("Randomization res")
for model in (Path(exp_path) / "imagenet").iterdir():  # Adjust the path as needed
    if model.stem == "data":
        continue
    # Loop through each method within the model
    for method in model.iterdir():
        res_path = method / "randomization.npy"
        if not res_path.exists():
            #   print(f"Skipping {res_path} as it does not exist")
            continue
        data = np.load(res_path)
        res.append((model.stem, method.stem, np.nanmean(data)))
res = pd.DataFrame(res, columns=["model", "method", "data"])
res = res[res.method.isin(main_methods)] if only_main else res
average_res = res.groupby("method").data.agg(["mean", "count"]).reset_index()
print(average_res.to_markdown(index=False))


print("normal res with standard deviation")
exps = [
    # "complexity.npy",
    "faithfullness.npy",
    # "sparseness.npy",
     "robustness.npy",
     "randomization.npy",
    "time.npy",
]

res = []

# Loop through each experiment
for exp_res in exps:
    for model in (Path(exp_path) / "imagenet").iterdir():  # Adjust the path as needed
        if model.stem == "data":
            continue
        # Loop through each method within the model
        for method in model.iterdir():
            res_path = method / exp_res
            if not res_path.exists():
                # print(f"Skipping {res_path} as it does not exist")
                continue

            data = np.load(res_path)
            if not (
                len(data.shape) == 2
                or res_path.stem == "time"
                or res_path.stem == "robustness"
                or res_path.stem == "randomization"
            ):
                # print(f"Skipping {res_path} as it does not have the correct shape or category")
                continue

            data = (
                data[0]
                if (
                    res_path.stem == "time"
                    or res_path.stem == "robustness"
                    or res_path.stem == "randomization"
                )
                else data[:, 1].astype(float)
            )
            # Append experiment, model, method, and the mean & std of the data
            res.append(
                (
                    model.stem,
                    method.stem,
                    res_path.stem,
                    np.nanmean(data),
                    np.nanstd(data),
                    data
                )
            )


# Create DataFrame
res_df = pd.DataFrame(res, columns=["model", "method", "experiment", "mean", "std","data"])


# Handling Machine Precisions problems:
#mask_from = (res_df.method == "ranem") & (res_df.experiment == "faithfullness")
#mask_to = (res_df.method == "ranem_ranked") & (res_df.experiment == "faithfullness")

#res_df.loc[mask_from, ["mean", "std"]] = res_df.loc[mask_to, ["mean", "std"]].values

# Now calculate the mean and std grouped by experiment and method
res_df_mean = (
    res_df.groupby(["experiment", "method"])[["mean", "std"]].agg("mean").reset_index()
)

# Optionally, you can add a row for the overall average by adding the following:
res_df_mean["model"] = "average"

res_df = pd.concat([res_df, res_df_mean]).reset_index(drop=True)
res_df["mean_std"] = (
    res_df["mean"].round(3).astype(str) + " ± " + res_df["std"].round(3).astype(str)
)

res_df["mean_std"] = res_df.apply(
    lambda row: row["mean_std"].split(" ")[0] if row["std"] == 0 else row["mean_std"],
    axis=1,
)

res_df = res_df.drop(columns=["mean", "std"])
res_df = res_df[res_df.method.isin(main_methods)] if only_main else res_df
# Create columns for each experiment
res_df = res_df.pivot(
    index=["model", "method"], columns="experiment", values="mean_std"
).reset_index()

print(res_df)

res_df = res_df[
    [
        "model",
        "method",
        "faithfullness",
        "robustness",
        "randomization",
        "time",
    ]
]
res_df.columns = [
    "model",
    "Method",
    "Faithfulness $\\uparrow$",
    "Robustness $\\downarrow$",
    "Randomization $\\downarrow$",
    "Time $\\downarrow$",
]


order = [
    "exp_rise",
    "exp_gradcam",
    "exp_intgrad",
    "exp_smoothpixelmask",
    "exp_gradshap",
    "exp_iba",
    "exp_saliency",
    "exp_nemt",
    "exp_ranem",
    "exp_intgrad_forgrad",
    "exp_saliency_forgrad",
    "exp_gradshap_forgrad",
]
res_df["Method"] = pd.Categorical(res_df["Method"], categories=order, ordered=True)
for model in res_df.model.unique()[::-1]:
    print("")
    print(model)
    print("")
    print(
        res_df[res_df["model"] == model]
        .drop(columns="model")
        .sort_values("Method")
        .to_markdown(index=False)
    )


if not show_significance:
    sys.exit()

print("")
print("Statistical significance faithfulness")
print("")

res_df = pd.DataFrame(res, columns=["model", "Method", "experiment", "mean", "std","data"])
methods =  res_df.Method.unique().astype(str)
models = res_df.model.unique().astype(str)
popped_element = None
#print(res_df)
#rank_index = np.where(np.char.find(methods, '_rank') >= 0)[0]
#assert len(rank_index) == 1, "Only one method with ranked attributions is allowed"
ranem_ranked = "exp_ranem"
methods = methods[methods != ranem_ranked]
faith_res = []
res_df = res_df[res_df.experiment == "faithfullness"]
for model in models:
    for method in main_methods:
        if method == ranem_ranked:
            continue
        try:
            sample2 = res_df[(res_df["Method"] == method) & (res_df["model"] == model)].data.values[0]
            sample1 = res_df[(res_df["Method"] == ranem_ranked) & (res_df["model"] == model)].data.values[0]
            stat, p_value = wilcoxon(sample1, sample2, alternative='two-sided')
            #print(f"{model} {method} vs {ranem_ranked}: {p_value}")
            faith_res.append([model, method, p_value])
            print(f"{model} {method} vs {ranem_ranked}: {p_value}")
        except:
            print(f"Skipping {model} {method} vs {ranem_ranked} due to missing data")
            continue
for method in main_methods:
    if method == ranem_ranked:
        continue
    if method == "exp_iba":
        res_df = res_df[res_df.model != "vit_base_patch16_224"]
    try:
        sample2 = res_df[(res_df["Method"] == method)].data.values
        sample2 = [x for xs in sample2 for x in xs ]
        sample1 = res_df[(res_df["Method"] == ranem_ranked)].data.values
        sample1 = [x for xs in sample1 for x in xs ]
        stat, p_value = wilcoxon(sample1, sample2, alternative='two-sided')
        print(f"Overall {method} vs {ranem_ranked}: {p_value}")
        #print(f"ALL {method} vs {ranem_ranked}: {p_value}")
        faith_res.append(["ALL", method, p_value])
    except:
        print(f"Skipping Overall {method} vs {ranem_ranked} due to missing data")
        continue

faith_res = pd.DataFrame(faith_res, columns=["model", "method", "p_value"])
faith_res = faith_res.sort_values(by=["model", "p_value"], ascending=[True, False])
faith_res["significant ( p < 0.05)"] = faith_res["p_value"] < 0.05
faith_res["significant ( p < 0.01)"] = faith_res["p_value"] < 0.01
faith_res["significant ( p < 0.001)"] = faith_res["p_value"] < 0.001

print(" All significance results")
print("")
print(faith_res)
print(" Not significant results p < 0.05")
print("")
print(faith_res[~faith_res["significant ( p < 0.05)"]])
print(" Not significant results p < 0.01")
print("")
print(faith_res[~faith_res["significant ( p < 0.01)"]])
print("")
print(" Not significant results p < 0.001")
print(faith_res[~faith_res["significant ( p < 0.001)"]])