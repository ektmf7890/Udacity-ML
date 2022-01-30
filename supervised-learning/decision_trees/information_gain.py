import pandas as pd
import math

def information_gain(data, col, class_name, threshold):
    mobug = data.loc[data["Species"] == "Mobug"].shape[0]
    lobug = data.loc[data["Species"] == "Lobug"].shape[0]
    p1 = mobug / (mobug + lobug)
    p2 = lobug / (mobug + lobug)
    entorpy_parent = - p1 * math.log2(p1) - p2 * math.log2(p2)
    # print(entorpy_parent)
    
    if class_name:
        group1 = data.loc[data[col] == class_name]
        group2 = data.loc[data[col] != class_name]
        

    elif threshold:
        group1 = data.loc[data[col] < threshold]
        group2 = data.loc[data[col] >= threshold]

    group1_mobug = group1.loc[group1["Species"] == "Mobug"].shape[0]
    group1_lobug = group1.loc[group1["Species"] == "Lobug"].shape[0]
    p1 = group1_mobug / (group1_mobug + group1_lobug)
    p2 = group1_lobug / (group1_mobug + group1_lobug)
    group1_entropy = - p1 * math.log2(p1) - p2 * math.log2(p2)

    group2_mobug = group2.loc[group2["Species"] == "Mobug"].shape[0]
    group2_lobug = group2.loc[group2["Species"] == "Lobug"].shape[0]
    p1 = group2_mobug / (group2_mobug + group2_lobug)
    p2 = group2_lobug / (group2_mobug + group2_lobug)
    group2_entropy = -p1 * math.log2(p1) - p2 * math.log2(p2)

    g17_ent = (group1_mobug+group1_lobug)/(mobug + lobug) * group1_entropy + (group2_mobug+group2_lobug)/(mobug+lobug) * group2_entropy
    return entorpy_parent - g17_ent 


if __name__ == "__main__":
    data = pd.read_csv("ml-bugs.csv", sep=',', header=0)
    print("Color=Brown: ", information_gain(data, "Color", "Brown", None))
    print("Color=Blue: ", information_gain(data, "Color", "Blue", None))
    print("Color=Green: ", information_gain(data, "Color", "Green", None))
    print("Length<17.0: ", information_gain(data, "Length (mm)", None, 17.0))
    print("Length<20.0: ", information_gain(data, "Length (mm)", None, 20.0))
        