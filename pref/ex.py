L = [ -6, -5 , -3, 0, 1, 1, 1, 2, 2, 3, 4, 6]


def minint(L):
    L = sorted(L)
    current = 1
    i = 0
    while i < len(L):
        if(L[i] <= 0):
            i = i + 1
            continue
        if(L[i] == current):
            while(i < len(L) and L[i] == current):
                i = i + 1
            current = current + 1
        else:
            return current

    return current

print(minint(L))


