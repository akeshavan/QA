import pandas as pd
import os
from glob import glob

#freesurfers


images = glob("*/*/*.png")

df = pd.DataFrame(data=images,columns=["image"])

df["subject"] = df.image.map(lambda x: os.path.split(x)[1].split("-")[0])
df["scanner"] =  df.image.map(lambda x: os.path.split(x)[1].split("-")[1] + "-" + os.path.split(x)[1].split("-")[3] + "-" +  os.path.split(x)[1].split("-")[5].split("_")[0])
df["type"] = df.image.map(lambda x: os.path.split(x)[0].split("/")[0])
df["view"] = df.image.map(lambda x: os.path.split(x)[1].split("_")[-1][0])

df.to_csv("freesurferqa.csv")

