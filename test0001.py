import random

a, b = input().split()
c, d = input().split()
e, f = input().split()
g, h = input().split()
i, j = input().split()

at = random.randint(1, 2)
bt = 3 - at
ct = random.randint(1, 2)
dt = 3 - ct
et = random.randint(1, 2)
ft = 3 - et
gt = random.randint(1, 2)
ht = 3 - gt
it = random.randint(1, 2)
jt = 3 - it

player_names = [a, b, c, d, e, f, g, h, i, j]
team_indexs = [at, bt, ct, dt, et, ft, gt, ht, it, jt]

print("1팀")
for i in range(0,10):
    if team_indexs[i] % 2 == 1:
        print(player_names[i])

print("2팀")
for i in range(0,10):
    if team_indexs[i]% 2 == 0:
        print(player_names[i])
