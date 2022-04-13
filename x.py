from urllib.request import urlopen
import json
from matplotlib import pyplot as plt
import numpy
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

url = "" #your json data link

response = urlopen(url)

data_json = json.loads(response.read())

veriler = data_json['data']

uzunluk = len(veriler)

y_data = []
x_data = []

for i in range(0,uzunluk):
    y_data.append(veriler[i][1])
    x_data.append(i)

lin = LinearRegression()
x_data = numpy.array(x_data)
y_data = numpy.array(y_data)
x_data = x_data.reshape(-1,1)
lin.fit(x_data, y_data)
 
poly = PolynomialFeatures(degree = 6)
X_poly = poly.fit_transform(x_data)
poly.fit(X_poly, y_data)

lin2 = LinearRegression()
lin2.fit(X_poly, y_data)

plt.scatter(x_data, y_data, color = 'blue')
 
plt.plot(x_data, lin2.predict(poly.fit_transform(x_data)), color = 'red')
plt.title('Polynomial Regression')
plt.xlabel('FİYAT')
plt.ylabel('GÜN')
 
plt.show()

for p in range(uzunluk,uzunluk+30):
    pred = p 
    predarray = numpy.array([[pred]])
    print(lin.predict(predarray))   



