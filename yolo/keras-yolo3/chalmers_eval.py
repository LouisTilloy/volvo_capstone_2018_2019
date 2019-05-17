import os
import ast
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib import rc
from cycler import cycler
import cv2

from chalmers_eval_utils import *

prop_cycle_test = cycler("color", ["r", "g", "b"])
prop_cycle_model = cycler("color", ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]) + cycler("marker", [".", "o", "8", "s", "p", "P", "*", "h", "+", "x"])

classes_path = "model_data/lisa_classes.txt"

"""
Define the input directory for the evaluation it should have the structure
model_name/
├── logs/
│   ├── x.tfevents
│   ├── y.tfevents
│   └── ...
└── trained_weights_final.h5
"""
eval_input_dir_v1 = "eval_input/Iteration 1"
eval_input_dir_v2 = "eval_input/Iteration 2"
eval_input_dir_v3 = "eval_input/Iteration 3"

# Define a directory {name:path} for the result outputs of the various YOLO testing sessions.
# NOTE: These result .csv-files need to be created first using the test_models() method.
# The reason we create these files is in order to perform efficent bootstrapping on the performance of YOLO.
# Instead of sampling random images many times and letting YOLO detect them, we can simply let YOLO first detect
# ALL images in the test set and collect the accuracy information in these .csv files. We can then simply sample
# randomly from this file for the bootstrap. This takes WAY less time compared to letting YOLO detect the images all the times.
eval_output_dir_v1 = "eval_output/Iteration 1"
eval_output_dir_v2 = "eval_output/Iteration 2"
eval_output_dir_v3 = "eval_output/Iteration 3"

# Define a directory {name:path} for the .txt files containing the various tests to perform bootstrapping on.
# When we bootstrap we will simply sample these entries randomly from the result files we created before.
eval_test_dict_v1 = {
        "Day" : "D:/Files/Dropbox/School/EENX15-19-21/datasets/lisa2012_test.txt",
        "Night" : "D:/Files/Dropbox/School/EENX15-19-21/datasets/bddnex.txt"
    }
eval_test_dict_v2 = {
        "Day" : "D:/Files/Dropbox/School/EENX15-19-21/datasets/lisa2012_test_c256.txt",
        "Night" : "D:/Files/Dropbox/School/EENX15-19-21/datasets/bddnex_c256.txt"
    }
eval_test_dict_v3 = eval_test_dict_v2

# Read the log files from the trained YOLO output and plot the losses
def eval_losses(in_dir):
    log_dict = {d : os.path.join(in_dir, d, "logs") for d in os.listdir(in_dir)}
    plot_losses(log_dict)

# Test the trained YOLO models on the .txt files containing all tests and save the results to a .csv file
def test_models(input_dir, test_paths_dict, image_size):
    model_dict = {d : os.path.join(input_dir, d, "trained_weights_final.h5") for d in os.listdir(input_dir)}
    # Load all tests to a single dict
    test_all_dict = {}
    for path in test_paths_dict.values():
        test_dict = load_data(path)
        test_all_dict = {**test_all_dict, **test_dict}

    for model_name, model_path in tqdm(model_dict.items(), desc="Model", leave=True, position=2):
        yolo = YOLOPlus(model_path=model_path, classes_path=classes_path, model_image_size=image_size)
        df = detect_result(test_all_dict, yolo)
        df_name = os.path.join("eval_output", os.path.basename(input_dir), model_name + ".csv")

        directory = os.path.dirname(df_name)
        if not os.path.exists(directory):
            os.makedirs(directory)

        df.to_csv(df_name, encoding="utf-8", index=True)

def eval_mistakes(eval_dir, test_path_dict):
    csv_dict = {}
    for f in os.listdir(eval_dir):  
        if f.endswith(".csv"):
            f_name = os.path.splitext(f)[0]
            split = f_name.split("_")
            if (split[0].isdigit()):
                f_name = "_".join(split[1:])
            csv_dict[f_name] = os.path.join(eval_dir, f)
    
    models = list(csv_dict.keys())

    df_res = pd.DataFrame(dtype=object)
    
    for test, path in test_path_dict.items():
        test_dict = load_data(path)
        for img, truths in test_dict.items():
            df_res = df_res.append({"test" : test, "image" : img, "truths" : truths, "mode" : None}, ignore_index=True)
    
    df_res.set_index("image", inplace=True, drop=True)

    for model, result_path in zip(models, csv_dict.values()):
        df_model = pd.read_csv(result_path, delimiter=",", encoding="utf-8")
        df_model.set_index("path", inplace=True, drop=True)
        df_model.rename(columns={"predictions": model}, inplace=True)
        df_res = df_res.join(df_model[model])
    
    df_res.reset_index(inplace=True)

    iou_threshold= 0.5
    confidence_threshold= 0.4
    
    timestamp = get_timestamp()

    N = len(df_res.index)
    for index, row in df_res.iterrows():
        print("Image %i of %i" % (index+1, N))
        truths = row.truths
        truths_results = []

        for model in models:
            predictions = ast.literal_eval(row[model])
            
            best_matching_prediction_indices = [-1]*len(truths)
            truth_results = [5]*len(truths)
            best_matching_truth_indices = [-1]*len(predictions)
            prediction_results = [5]*len(predictions)
            
            # [best matching pred index]
            # [truth res]
            # [best matching truth index]
            # [pred res]
            # [x0, y0, x1, y1, class, conf]

            # PREDICTION RESULTS
            # 0 if the prediction is considered True
            # 1 if the prediction is not confident enough
            # 2 if the prediction has right class and low IoU
            # 3 if the prediction has wrong class and ok IoU
            # 4 if the prediction has the wrong class and low IoU
            # 5 if the prediction has no IoU

            # TRUTHS
            # -1 if no truth was found
            # PREDICTION_INDEX if the truth was not found with the prediction with index PREDICTION_INDEX
            
            for i, p in enumerate(predictions):
                for j, t in enumerate(truths):
                    iou = IoU(p[0:4], t[0:4])
                    iou_present = iou > 0
                    iou_ok = iou >= iou_threshold
                    class_ok = p[4] == t[4]
                    confidence_ok = p[5] >= confidence_threshold
                    
                    if iou_ok and class_ok and confidence_ok:
                        prediction_results[i] = 0
                        best_matching_truth_indices[i] = j
                    elif iou_ok and class_ok and not confidence_ok and prediction_results[i] > 1:
                        prediction_results[i] = 1
                        best_matching_truth_indices[i] = j
                    elif iou_ok and not class_ok and prediction_results[i] > 2:
                        prediction_results[i] = 2
                        best_matching_truth_indices[i] = j
                    elif iou_present and not iou_ok and class_ok and prediction_results[i] > 3:
                        prediction_results[i] = 3
                        best_matching_truth_indices[i] = j
                    elif iou_present and not iou_ok and not class_ok and prediction_results[i] > 4:
                        prediction_results[i] = 4
                        best_matching_truth_indices[i] = j
                    
                    if best_matching_truth_indices[i] == j and truth_results[j] > prediction_results[i]:
                        best_matching_prediction_indices[j] = i
                        truth_results[j] = prediction_results[i]
            
            truths_results.append(truth_results)

            df_res.loc[index, model] = [best_matching_prediction_indices, truth_results, best_matching_truth_indices, prediction_results, predictions]

        mode = [get_mode([res[i] for res in truths_results]) for i in range(len(truths))]
        df_res.loc[index, "mode"] = mode
    
    df_res_name = "eval_output/" + os.path.basename(eval_dir) + "/pass_fail/eval_" + timestamp + "/eval_img.csv"
    os.makedirs(os.path.dirname(df_res_name), exist_ok=True)
    df_res.to_csv(df_res_name, encoding="utf-8", index=False)

    cols = ["test", "image", "x0", "y0", "x1", "y1", "c", "mode"] + models
    df_ex = pd.DataFrame(columns=cols, dtype=object)
    for index, row in df_res.iterrows():
        for i, t in enumerate(row.truths):
            results = [row[model][1][i] for model in models]
            new_row = pd.Series([row.test, row.image, t[0], t[1], t[2], t[3], t[4], row["mode"][i]] + results, cols)
            df_ex = df_ex.append([new_row], ignore_index=True)

    df_res_name = "eval_output/" + os.path.basename(eval_dir) + "/pass_fail/eval_" + timestamp + "/eval_ex.csv"
    os.makedirs(os.path.dirname(df_res_name), exist_ok=True)
    df_ex.to_csv(df_res_name, encoding="utf-8", index=False)

    print(df_ex)
    
# After we have generated the results .csv files, we can use the tests defined in test_bootstrap_dict to perform the bootstrap tests.
# We will also create different plots for the different models we have trained.
def eval_performance(eval_dir, test_path_dict, plot_range):
    csv_dict = {}
    for f in os.listdir(eval_dir):  
        if f.endswith(".csv"):
            f_name = os.path.splitext(f)[0]
            split = f_name.split("_")
            if (split[0].isdigit()):
                f_name = "_".join(split[1:])
            csv_dict[f_name] = os.path.join(eval_dir, f)

    test_dict = {name : list(load_data(path).keys()) for name, path in test_path_dict.items()}
    test_dict["All"] = [path for test in test_dict.values() for path in test]

    sample_ratio = 1.0
    B = 1000
    seed = 42
    iou_threshold= 0.5

    df_res = pd.DataFrame(columns=["order", "model", "test", "pmu", "pstd", "rmu", "rstd", "N", "n", "B", "p", "r"], dtype=object)

    for i, (model, result_path) in enumerate(csv_dict.items()):
        df = pd.read_csv(result_path, delimiter=",", encoding="utf-8")
        df.set_index("path", inplace=True, drop=True)
        
        for test, imgs in test_dict.items():
            N = len(imgs)
            n = int(sample_ratio * N)
            np.random.seed(seed)

            precision = []
            recall = []

            for _ in tqdm(range(B), desc="Iteration", leave=True, position=2):
                indices = np.random.randint(N, size=n)
                sample_imgs = [imgs[i] for i in indices]
                
                total_truths = 0
                total_predictions = 0
                total_correct = 0

                for img in sample_imgs:
                    truths = ast.literal_eval(df.at[img, "truths"])
                    n_truths = len(truths)
                    total_truths += n_truths

                    predictions = ast.literal_eval(df.at[img, "predictions"])
                    n_predictions = len(predictions)
                    total_predictions += n_predictions
                    
                    for p in predictions:
                        for t in truths:
                            intersect_ok = IoU(p[0:4], t[0:4]) > iou_threshold
                            class_ok = p[4] == t[4]
                            if (intersect_ok and class_ok):
                                total_correct += 1
                                break

                precision.append(total_correct / total_predictions)
                recall.append(total_correct / total_truths)
                precision.sort()
                recall.sort()

            pmu, pstd = stats.norm.fit(precision)
            rmu, rstd = stats.norm.fit(recall)

            df_res = df_res.append({"order" : i, "model" : model, "test" : test, "pmu" : pmu, "pstd" : pstd, "rmu" : rmu, "rstd" : rstd, "N" : N, "n" : n, "B" : B, "p" : precision, "r" : recall}, ignore_index=True)
    
    df_res_name = "eval_output/" + os.path.basename(eval_dir) + "/evaluations/eval_" + get_timestamp() + ".csv"
    os.makedirs(os.path.dirname(df_res_name), exist_ok=True)
    df_res.to_csv(df_res_name, encoding="utf-8", index=False)

# Plot the Probability Density Function and the Histograms of the resulting bootstrapping tests.
# plot_data contains the data from all models on all tests.
def plot_pdf(df_name, plot_range):
    df = pd.read_csv(df_name, delimiter=",", encoding="utf-8")
    
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
            
    models = df.model.unique()
    size = (8, 4)

    base = df.loc[df.order == 0]
    base_m = base.model.values[0]
    base_p = dict(zip(base.test, [ast.literal_eval(x) for x in base.p]))
    base_r = dict(zip(base.test, [ast.literal_eval(x) for x in base.r]))

    #for model in models:
    #    plot_pdf_curves(df, "Precision", "p", model, base_m, base_p, plot_range, size)
    #    plot_pdf_curves(df, "Recall", "r", model, base_m, base_r, plot_range, size)

    plot_pdf_box("Precision", "p", df, models, plot_range, size)
    plot_pdf_box("Recall", "r", df, models, plot_range, size)

    plt.show()

def plot_pdf_curves(df, name, value, model, base_m, base_v, plot_range, size):
    data = df.loc[df.model == model]
    plt.figure(figsize=size)
    plt.gca().set_prop_cycle(prop_cycle_test)
    x = np.linspace(plot_range[0], plot_range[1], 1000)
    x_tick = 0.1

    for test in data.test.unique():
        values = data.loc[df.test == test].iloc[0]
        v = ast.literal_eval(values[value])
        N = values.N
        n = values.n
        B = values.B

        mu, std = stats.norm.fit(v)
        pdf = stats.norm.pdf(x, mu, std)
        lbl = r"%s: $\mu = %.3f, \sigma$ = %.3f, N = %i, n = %i, B = %i" % (test, mu, std, N, n, B)
        curve, = plt.plot(x, pdf, linewidth=2, label=lbl)

        if model != base_m:
            v0 = base_v[test]
            mu0, std0 = stats.norm.fit(v0)
            pdf0 = stats.norm.pdf(x, mu0, std0)
            plt.plot(x, pdf0, linewidth=2, color=curve.get_color(), linestyle=":", alpha=0.6)

        plt.hist(v, bins=25, density=True, color=curve.get_color(), alpha=0.4)
        plt.xticks(np.arange(plot_range[0], plot_range[1] + x_tick, x_tick))
        plt.xlim(plot_range)
    
    plt.grid()
    plt.legend()
    plt.gca().set(title="Bootstrap %s of %s" % (name, get_latex(model)), ylabel="PDF", xlabel=name)

def plot_pdf_box(name, value, df, models, plot_range, size):
    fig, axes = plt.subplots(ncols=len(models), sharey=True, figsize=size)
    fig.subplots_adjust(wspace=0)
    fig.suptitle("Bootstrap " + name)
    axes[0].set_ylabel(name)
    fig.subplots_adjust(top=0.9)

    for ax, model in zip(axes, models):
        data = df.loc[df.model == model]
        bp = ax.boxplot([ast.literal_eval(x) for x in data[value]], patch_artist=True)
        colors = prop_cycle_test.by_key()["color"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
        plt.ylim(plot_range)
        ax.set_xlabel(get_latex(model), fontsize = 9, rotation=0)
        ax.set(xticklabels=data.test.values)
        plt.setp(ax.get_xticklabels(), fontsize=9, rotation=45)
        plt.subplots_adjust(bottom=0.15)
        ax.yaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.5)

# Plot the loss curves during training.
def plot_losses(logs_dict):
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
            
    plt.figure(1, figsize=(8,4))
    for name, path in logs_dict.items():
        log = load_log_dir(path)
        x = [item["step"] for item in log]
        y = [item["val_loss"] for item in log]
        
        #Smooth out missing data points
        for j in reversed(range(0, len(y))):
            if (y[j] == 0.0):
                y[j] = y[j+1]

        plt.semilogy(x, y, linewidth=2, label=get_latex(name))
        plt.gca().set(title="Validation loss of YOLO", ylabel="Validation loss", xlabel="Epoch")
    
    plt.grid()
    plt.legend()
    plt.show()

def detect_result(dic, yolo):
    imgs = list(dic.keys())
    df = pd.DataFrame(columns=["path", "truths", "predictions"], dtype=object)
    
    for img in imgs:
        df = df.append({"path" : img, "truths" : dic[img], "predictions" : []}, ignore_index=True)

    df.set_index("path", inplace=True, drop=True)

    for img in tqdm(imgs, desc="Images", leave=True, position=2):
        labels, scores, boxes = detect_img(img, yolo)
        for box, label, score in zip(boxes, labels, scores):
            l = [box[1], box[0], box[3], box[2], label, score]
            df.at[img, "predictions"].append(l)
    
    return df

def eval_distribution(*paths, lookup=None, names=None):
    dicts = [load_data(p) for p in paths]
    ann = {get_latex(p) : [t[4] for truths in d.values() for t in truths] for p, d in zip(names if names != None else paths, dicts)}
    n_ann = [len(l) for l in ann.values()]
    n_ann_a = sum(n_ann)

    classes_max = max([v for array in ann.values() for v in array])
    n_classes = classes_max + 1

    sums = np.zeros(n_classes)
    for array in ann.values():
        for v in array:
            sums[v] += 1

    if lookup != None:
        with open(lookup, "r") as f:
            lines = f.readlines()
            lines = [l.rstrip() for l in lines]
            #r = ["%s (%i)" % (lines[i], i) for i in range(0, classes_max)]
            r = ["%s (%i)" % (lines[i], sums[i]) for i in range(n_classes)]
    else:
        r = range(0, n_classes)


    dists = {}
    for p, array in ann.items():
        hist = [0] * n_classes
        for v in array:
            hist[v] += 1
        #hist = [x for _, x in sorted(zip(sums, hist))]
        dists[p] = hist

    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    size = (8,8)

    """
    fig = plt.figure(1, figsize=size)
    #plt.gca().set_prop_cycle(prop_cycle_test)
    for (p, l) in ann.items():
        #dist, _ = np.histogram(l, density=False, bins=classes_max, range=(0, classes_max))
        #dists[p] = dist
        #dist = [x for _, x in sorted(zip(sums, dist))]
        #y = sums.sort()
        plt.barh(r, sums, edgecolor="white", alpha=0.6, align="center", label=p)
    plt.gca().set(title="Occurances", ylabel="Class", xlabel="Occurances")
    plt.legend()
    plt.yticks(r)
    plt.gca().xaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.5)
    plt.setp(plt.gca().get_yticklabels(), fontsize=6, rotation=0)
    fig.tight_layout()
    """

    exs = {p : [] for p in ann.keys()}
    for c in range(0, n_classes):
        occ = [dists[p][c] > 0 for p in ann.keys()]
        if sum(occ) == 1:
            index = next((i for i, j in enumerate(occ) if j), None)
            exs[list(ann.keys())[index]].append(c)

    n_ex = {}
    for p, l in exs.items():
        n_ex[p] = sum([dists[p][c] for c in l])
    
    dist_all = [sum([dists[p][c] for p in ann.keys()]) for c in range(0, n_classes)]

    ratios = {}
    
    for p in ann.keys():
        p_ratios = []
        for h, ha in zip(dists[p], dist_all):
            p_ratios.append(h / ha if ha != 0 else 0)
        ratios[p] = p_ratios
    
    indices = np.argsort(sums)
    r = [r[i] for i in indices]
    for k in ann.keys():
        ratios[k] = [ratios[k][i] for i in indices]
    #r = [x for _, x in sorted(zip(sums, r))]
    #ratios = {k : [x for _, x in sorted(zip(sums, ratios[k]))] for k in ratios.keys()}
    #r = [x for _, x in sorted(zip(sums, r))]

    #sums.sort()

    fig = plt.figure(2, figsize=size)
    #plt.gca().set_prop_cycle(prop_cycle_test)
    prev_ratio = 0.0
    for p, ratio in ratios.items():
        plt.barh(r, ratio, left=prev_ratio, edgecolor="white", height=0.85, label=p)
        prev_ratio = ratio
    plt.yticks(r)
    plt.setp(plt.gca().get_yticklabels(), fontsize=6, rotation=0)
    plt.legend()
    plt.gca().set(title="Ratio of classes sorted by total number of occurances", ylabel="Class (total occurances)", xlabel="Ratio")
    fig.tight_layout()

    for p in ann.keys():
        print()
        print("Classes exclusive to %s:" % p)
        print(exs[p])
        print("Number of occurances exclusive to %s:" % p)
        print("%i occurances of %i. (%.3f%%)" % (n_ex[p], n_ann_a, 100 * n_ex[p] / n_ann_a))

    plt.show()

def save_images(path, lookup=None):
    df = pd.read_csv(path, delimiter=",", encoding="utf-8")
    model_names = [x for x in list(df.columns.values) if x not in ["image", "mode", "test", "truths"]]
    N = len(df.index)

    timestamp = get_timestamp()
    class_names = load_classes(classes_path)
    colors = get_colors(class_names)

    for index, row in df.iterrows():
        print("Image %i of %i" % (index+1, N))
        for model in model_names:
            mode = ast.literal_eval(row["mode"])
            m_res = ast.literal_eval(row[model])
            m_pred_res = m_res[3]
            m_pred = m_res[4]
            b_res = ast.literal_eval(row[model_names[0]])
            b_pred_res = b_res[3]
            b_pred = b_res[4]

            for (m, mr, mpr, mp, br, bpr, bp) in zip(mode, m_res, m_pred_res, m_pred, b_res, b_pred_res, b_pred):
                if mpr < m:
                    save_dir = os.path.dirname(path) + "/pass_fail/eval_" + timestamp + "/" + model + "/above_mode/" + df.loc[index, "test"]
                    img = df.loc[index, "image"]
                    save_image(img, save_dir, class_names, colors, [mp])
                elif mpr > m:
                    save_dir = os.path.dirname(path) + "/pass_fail/eval_" + timestamp + "/" + model + "/below_mode/" + df.loc[index, "test"]
                    img = df.loc[index, "image"]
                    save_image(img, save_dir, class_names, colors, [mp])                

def plot_mistakes(path, lookup=None):
    df_src = pd.read_csv(path, delimiter=",", encoding="utf-8")
    model_names = [x for x in list(df_src.columns.values) if x not in ["test", "image", "x0", "y0", "x1", "y1", "c", "mode"]]
    mistake_strings = ["Low confidence", "Wrong class", "Low IoU", "Wrong class and low IoU", "No IoU"]
    
    test_names = list(df_src.test.unique())
    test_sizes = {test : 0 for test in test_names}
    for test in df_src.test.values:
        test_sizes[test] += 1

    new_column_names = ["c", "n"] + model_names
    df_cls = pd.DataFrame(columns=new_column_names, dtype=object)
    df_cls.set_index("c", inplace=True, drop=True)
    
    classes = sorted(list(df_src.c.unique()))

    for c in classes:
        rows = df_src[df_src.c == c]
        df_cls.loc[c, "n"] = len(rows)
        for model in model_names:
            results = rows[model].tolist()
            df_cls.loc[c, model] = sum([1 if r == 0 else 0 for r in results])

    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")

    s = df_cls[model_names].sum(axis=1, skipna=True)
    ave = s / len(model_names)
    df_cls["ave"] = ave
    ave_ratio = ave / df_cls.n
    df_cls["ave_ratio"] = ave_ratio
    df_cls = df_cls.sort_values(by="ave_ratio", ascending=True)

    plot_detections_by_class(False, df_cls, lookup, model_names, classes, (8,5))
    plot_detections_by_class(True, df_cls, lookup, model_names, classes, (8,5))
    plot_mistakes_by_test(True, df_src, test_names, model_names, mistake_strings, (8,5))

    plt.show()

def plot_detections_by_class(ratio, df_cls, lookup, model_names, classes, size):
    fig = plt.figure(figsize=size)
    if ratio:
        df_cls = df_cls.sort_values(by="n", ascending=True)
    else:
        df_cls = df_cls.sort_values(by="n", ascending=True)

    plt.gca().set_prop_cycle(prop_cycle_model)
    for model in model_names:
        x = df_cls[model] / df_cls.n if ratio else df_cls[model]
        plt.plot(x.tolist(), range(len(classes)), markersize=4, linestyle="None", label=get_latex(model))
    
    plt.gca().set(title="Ratio of detected occurances by class and method sorted by total number of occurances" if ratio else "Number of detected occurances by class and method sorted by total number of occurances", ylabel="Class (total occurances)", xlabel="Ratio of detected occurances" if ratio else "Number of detected occurances")
    
    cls_labels = ["%i (%i)" % (c, n) for (c, n) in zip(classes, df_cls.n.values)]
    if lookup != None:
        with open(lookup, "r") as f:
            lines = f.readlines()
            lines = [l.rstrip() for l in lines]
            cls_labels = ["%s (%i)" % (lines[i], n) for (i, n) in zip(df_cls.index.values, df_cls.n.values)]

    plt.yticks(ticks=range(len(classes)), labels=cls_labels)
    plt.setp(plt.gca().get_yticklabels(), fontsize=6, rotation=0)
    plt.grid()
    plt.legend()
    fig.tight_layout()

def plot_mistakes_all(ratio, df_src, test_names, model_names, mistake_strings, size):
    fig = plt.figure(figsize=size)
    
    n = len(model_names)
    r = 1 + np.arange(len(mistake_strings))
    h = 0.05

    plt.gca().set_prop_cycle(prop_cycle_model)

    for i, model in enumerate(model_names):
        mistakes = list(filter(lambda x : x != 0, df_src[model].tolist()))
        sum_mistake = []
        for mistake_type in r:
            s = 0
            for m in mistakes:
                if m == mistake_type:
                    s += 1
            sum_mistake.append(s)

        if ratio:
            sum_mistake = [m / len(mistakes) for m in sum_mistake]

        plt.barh(r + (i-(n-1)/2)*h, sum_mistake, edgecolor="white", height=h, label=get_latex(model))
    
    plt.legend()
    plt.yticks(ticks=r, labels=mistake_strings)
    plt.gca().set(title="Ratio of mistakes by method" if ratio else "Number of mistakes by method", ylabel="Mistake", xlabel="Ratio of mistakes" if ratio else "Number of mistakes")
    plt.gca().xaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.5)
    fig.tight_layout()

def plot_mistakes_by_test(ratio, df_src, test_names, model_names, mistake_strings, size):
    fig, axes = plt.subplots(nrows=len(test_names), sharex=True, figsize=size)
    fig.subplots_adjust(hspace=0)
    fig.subplots_adjust(top=0.9)

    h = 0.08
    r = 1 + np.arange(len(mistake_strings))
    n = len(model_names)

    for i, (ax, test) in enumerate(zip(axes, test_names)):
        axes[i].set_prop_cycle(prop_cycle_model)

        for j, model in enumerate(model_names):
            mistakes = list(filter(lambda x : x != 0, df_src.loc[df_src["test"] == test, model]))
            sum_mistake = []
            for mistake_type in r:
                s = 0
                for m in mistakes:
                    if m == mistake_type:
                        s += 1
                sum_mistake.append(s)
            
            if ratio:
                sum_mistake = [m / len(mistakes) for m in sum_mistake]

            p, = ax.plot(sum_mistake, r + (j-(n-1)/2)*h, linestyle="", label=get_latex(model) if i == 0 else "")
            ax.barh(r + (j-(n-1)/2)*h, sum_mistake, color=p.get_color(), edgecolor="white", height=h)

        ax.set_ylabel(get_latex(test), fontsize = 10, rotation=0)
        ax.set(yticks=r)
        ax.set(yticklabels=mistake_strings)
        plt.setp(ax.get_yticklabels(), fontsize=8, rotation=0)
        ax.xaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.5)
    
    title = fig.suptitle(" Ratio of mistakes by test and method" if ratio else "Number of mistakes by test and method")
    plt.gca().set(xlabel="Ratio of mistakes" if ratio else "Number of mistakes")
    plt.subplots_adjust(left=0.25)
    plt.figlegend()
    fig.tight_layout()
    title.set_y(0.95)
    fig.subplots_adjust(top=0.90)

def comp_datasets(dataset_a, dataset_b=None, size=(4,4), scale=1.0):
    dict_images_a = {os.path.basename(key) : key for key in load_data(dataset_a).keys()}

    np.random.seed(3)
    H, W = size[0], size[1]
    n = size[0] * size[1]
    img_names = list(dict_images_a.keys())
    N = len(img_names)

    indices = np.random.randint(N, size=int(n if dataset_b is None else n/2))
    img_name_samples = [img_names[i] for i in indices]
    imgs_a = [np.uint8(cv2.imread(dict_images_a[name], 3)) for name in img_name_samples]

    if dataset_b is not None:
        dict_images_b = {os.path.basename(key) : key for key in load_data(dataset_b).keys()}
        imgs_b = [np.uint8(cv2.imread(dict_images_b[name], 3)) for name in img_name_samples]
    
    h, w = imgs_a[0].shape[0], imgs_a[0].shape[1]
    img = np.zeros((int(h*H), int(w*W), 3), np.uint8)

    for idx in range(n):
        i = idx % W
        j = idx // W
        if dataset_b is not None:
            insert = cv2.resize(imgs_a[idx // 2] if (idx % 2 == 0) else imgs_b[idx // 2], (h, w))
        else:
            insert = cv2.resize(imgs_a[idx], (h, w))

        img[j*h:(j+1)*h, i*w:(i+1)*w, :] = insert

    img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)

    name = os.path.splitext(os.path.basename(dataset_a))[0]
    if dataset_b is not None:
        name += "_" + os.path.splitext(os.path.basename(dataset_b))[0]

    os.makedirs("samples", exist_ok=True)
    cv2.imwrite("samples/" + name + ".png", img)

    #cv2.imshow(name, img)
    #cv2.waitKey(0)

def detect(imgs, model_path, classes_path, image_size):
    yolo = YOLOPlus(model_path=model_path, classes_path=classes_path, model_image_size=image_size)
    class_names = load_classes(classes_path)
    colors = get_colors(class_names)
    
    timestamp = get_timestamp()
    save_dir = "detections_" + timestamp

    for img in imgs:
        predictions = []
        labels, scores, boxes = detect_img(img, yolo)
        for box, label, score in zip(boxes, labels, scores):
            predictions.append([box[1], box[0], box[3], box[2], label, score])
        save_image(img, save_dir, class_names, colors, predictions)

def main():
    dataset_lisa = "C:/Users/Adam/Dropbox/School/EENX15-19-21/datasets/lisa2012_train_c256.txt"
    dataset_blend = "C:/Users/Adam/Dropbox/School/EENX15-19-21/datasets/blend256.txt"
    dataset_saug = "C:/Users/Adam/Dropbox/School/EENX15-19-21/datasets/saug2_lisa_c256.txt"
    dataset_cg = "C:/Users/Adam/Dropbox/School/EENX15-19-21/datasets/cgan2_lisa2012_train_c256.txt"
    dataset_cg_ins = "C:/Users/Adam/Dropbox/School/EENX15-19-21/datasets/cgan2_lisa2012_train_c256_ins.txt"
    dataset_bbg = "C:/Users/Adam/Dropbox/School/EENX15-19-21/datasets/bbgan1_ff_cnn_lisa2012_train_c256.txt"
    
    #test_models(eval_input_dir_v1, eval_test_dict_v1, (416, 416))
    #test_models(eval_input_dir_v2, eval_test_dict_v2, (256, 256))
    #test_models(eval_input_dir_v3, eval_test_dict_v3, (256, 256))
    #eval_performance(eval_output_dir_v1, eval_test_dict_v1, (0.0, 1.0))
    #eval_performance(eval_output_dir_v3, eval_test_dict_v3, (0.6, 1.0))
    #plot_pdf("eval_output/Iteration 1/evaluations/eval_205424.csv", (0.0, 1.0))
    #plot_pdf("eval_output/Iteration 2/evaluations/eval_194953.csv", (0.6, 1.0))
    #save_images("eval_output/Iteration 2/pass_fail/eval_165558/eval_img.csv")
    #eval_mistakes(eval_output_dir_v2, eval_test_dict_v2)
    #plot_mistakes("eval_output/Iteration 2/pass_fail/eval_192721/eval_ex.csv", lookup="lisa_classes.txt")
    eval_losses(eval_input_dir_v2)
    #eval_distribution("lisa2012_train_c256.txt", "lisa2012_test_c256.txt", names=["LISA train", "LISA test"], lookup=classes_path)
    #eval_distribution("lisa2012_train_c256.txt", "bddnex_c256.txt", names=["LISA train", "BDDNEX"], lookup=classes_path)
    #eval_distribution("lisa2012_test_c256.txt", "bddnex_c256.txt", names=["LISA test", "BDDNEX"], lookup=classes_path)

    #comp_datasets(dataset_blend, size=(6,4))
    #comp_datasets(dataset_lisa, dataset_b=dataset_saug, size=(6,4))
    #comp_datasets(dataset_lisa, dataset_b=dataset_cg, size=(6,4))
    #comp_datasets(dataset_lisa, dataset_b=dataset_cg_ins, size=(6,4))
    #comp_datasets(dataset_lisa, dataset_b=dataset_bbg, size=(6,4))
    
    #comp_datasets(dataset_blend, size=(1,4))
    #comp_datasets(dataset_lisa, dataset_b=dataset_saug, size=(1,4))
    #comp_datasets(dataset_lisa, dataset_b=dataset_cg, size=(1,4))
    #comp_datasets(dataset_lisa, dataset_b=dataset_cg_ins, size=(1,4))
    #comp_datasets(dataset_lisa, dataset_b=dataset_bbg, size=(1,4))

    #detect(["bbg_ff_cn_stop_1398812846.avi_image10.png", "stop_1398812846.avi_image10.png", "frame_6769de8ca0e1910691f3dc339e4d82db_82520a30e15767017b31462e7e40349c_60000-1280_720.png"], "eval_input/Iteration 2/AAUG_BBG_FF_CN/trained_weights_final.h5", classes_path, (256, 256))

if __name__ == "__main__":
    main()