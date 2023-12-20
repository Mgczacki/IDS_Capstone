import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from QuestionFour import data_prep


spotify52k = data_prep()
q3data = spotify52k[["mode", "popularity"]]


MajorPop = pd.value_counts(q3data[q3data["mode"] == 1].popularity.values, normalize=True)
MinorPop = pd.value_counts(q3data[q3data["mode"] == 0].popularity.values, normalize=True)
fig = plt.figure()
ax = plt.subplot()
fig.suptitle("Popularity barplot for Major & Minor key songs")
ax.bar(x=MajorPop.index, height=MajorPop, color="blue", label="Major")

fig.supylabel("Probability Density")
fig.supxlabel("Popularity Measure (Higher = More popular)")

ax.bar(x=MinorPop.index, height=MinorPop, color="orange", label="Minor", alpha=0.8)
fig.legend(frameon=False)

testStat, pvalue = mannwhitneyu(MinorPop, MajorPop)
print(f"pvalue ({pvalue}) is higher than our significance level 0.05, hence the null hypothesis (songs in either key"
      f"are equally is cannot be rejected.")
plt.savefig("q3ProbDensity.png")
plt.show()
