import numpy as np

i = 0
shapes = dict()
with open("np_arrays/happy_dog_div8 copy.npy", "rb") as f:
    while True:
        try:
            x = np.load(f)
            i += 1
            if x.shape in shapes:
                shapes[x.shape] += 1
            else:   
                shapes[x.shape] = 1
            print(x.mean())
        except:
            break

print('Total:')
print(i)
print(shapes)   