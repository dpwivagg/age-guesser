import numpy as np
faces_test = np.empty((0,48,48))
ages_test = np.empty((0))
faces_train = np.empty((0,48,48))
ages_train = np.empty((0))

faces = np.load('data/faces.npy')
ages = np.load('data/ages.npy')

for i in range(7500):
    rand = np.random.rand()
    if rand < 0.1:
        faces_test = np.append(faces_test, [faces[i]], axis=0)
        ages_test = np.append(ages_test, ages[i])
    else:
        faces_train = np.append(faces_train, [faces[i]], axis=0)
        ages_train = np.append(ages_train, ages[i])

np.save('data/ages_test.npy', ages_test)
np.save('data/ages_train.npy', ages_train)
np.save('data/faces_test.npy', faces_test)
np.save('data/faces_train.npy', faces_train)