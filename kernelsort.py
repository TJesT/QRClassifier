op = open("kernels.txt")
l = sorted(op.readlines(), key=lambda x: len(x))
l = sorted(l, key=lambda x: sum(map(int, x.split())))
op.close()

wr = open("kernels.txt", "w")
for r in l:
    wr.write(r)
wr.close()
