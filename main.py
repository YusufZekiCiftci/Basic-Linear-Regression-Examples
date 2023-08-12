import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv("csv/Advertising.csv")
df = df.iloc[:, 1:len(df)]
sns.jointplot(x="TV", y="sales", data=df, kind="reg")
plt.show()

X = df[["TV"]]
X.head()

Y = df[["sales"]]
Y.head()

reg = LinearRegression()
model = reg.fit(X, Y)

# rkare
sonuc = model.score(X, Y)

print(sonuc)
