
from math import pi, factorial, sqrt

def f(i, j, k):
    a = 0.800000
    return (2*a/pi)**0.75 * sqrt((8*a)**(i+j+k)*factorial(i)*factorial(j)*factorial(k)/(factorial(2*i)*factorial(2*j)*factorial(2*k)))

J = 2
i = J // 3
j = i + (J % 3) // 2
k = J - i - j
print(i, j, k)

for i in range(0, J+1):
    for j in range(0, J+1):
        k = J - i - j
        if 0 <= k <= J:
            print(i, j, k, f(i, j, k))