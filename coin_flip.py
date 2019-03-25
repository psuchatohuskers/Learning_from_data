import numpy as np

def flip_coin(coin_num,flip_time):
    head_count = []
    c = np.zeros(3)
    for coin in range(coin_num):
        coin_flip = np.random.randint(0,2,flip_time)
        head_count.append(np.mean(coin_flip))
    c[0] = head_count[0]
    c[1] = np.random.choice(head_count)
    c[2] = np.amin(head_count)
    return c


v_sim = []
for i in range(10000):
    v_sim.append(flip_coin(1000,10))
v_array = np.array(v_sim)
v_final = np.mean(v_array,axis = 0)
print(v_final)
