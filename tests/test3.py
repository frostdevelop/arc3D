import numpy as np
let = np.asarray([1,2,3,4,5,6,7,8,9,10]).astype(np.float64)
print(", ".join(map(str, let)))
print(*let)
print(*let, sep = ", ")
print(f"{*let,}")
print(", ".join(map(str, (let[i] for i in range(5)))))
print(np.zeros((3,2)))