import pandas as pd

df = pd.read_csv("/Users/andrewchurchill/Documents/WOD/2023_Men_info.csv")

# Keep copy of original dataframe
ogdf = df.copy()

# Create weight and height dataframes with the nan values dropped from each column
no_nan_weights_df = df.dropna(subset=["weight"])[["id", "weight"]]
no_nan_heights_df = df.dropna(subset=["height"])[["id", "height"]]

# Convert kg values to lb, these are only the values that ended in kg in the original df
kg_filter = no_nan_weights_df["weight"].str.endswith("kg")
new_df = no_nan_weights_df[kg_filter]
lb_df = new_df.copy()
lb_df["weight"] = lb_df["weight"].str[:-2]
lb_df["weight"] = (lb_df["weight"].astype(float) * 0.453592).round().astype(int).astype(
    str
) + " lb"

# Converting cm values to inches, these are only the values that ended in cm in the original df
cm_filter = no_nan_heights_df["height"].str.endswith("cm")
n_df = no_nan_heights_df[cm_filter]
in_df = n_df.copy()
in_df["height"] = in_df["height"].str[:-2]
in_df["height"] = (in_df["height"].astype(float) * 0.393701).round().astype(int).astype(
    str
) + " in"

# Adding converted values into the each df matching by id
no_nan_weights_df.loc[no_nan_weights_df["id"].isin(lb_df["id"]), "weight"] = lb_df[
    "weight"
]
no_nan_heights_df.loc[no_nan_heights_df["id"].isin(in_df["id"]), "height"] = in_df[
    "height"
]

# change into integer
no_nan_weights_df["weight"] = no_nan_weights_df["weight"].str[:-2].astype(int)
no_nan_heights_df["height"] = no_nan_heights_df["height"].str[:-2].astype(int)

# find the means of all the original and converted values combined
mean_weight = round(no_nan_weights_df["weight"].mean())
mean_height = round(no_nan_heights_df["height"].mean())

# Fill nan-values from original dataframe with the respective means
ogdf["weight"] = ogdf["weight"].fillna(str(mean_weight) + " lb")
ogdf["height"] = ogdf["height"].fillna(str(mean_height) + " in")
ogdf.loc[ogdf["id"].isin(lb_df["id"]), "weight"] = lb_df["weight"]
ogdf.loc[ogdf["id"].isin(in_df["id"]), "height"] = in_df["height"]

print(ogdf)
print("mean weight is " + str(mean_weight) + " pounds")
print("mean height is " + str(mean_height) + " inches")
