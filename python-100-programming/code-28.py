def age(n):
    if n == 1: Ag = 10
    else: Ag = age(n-1) + 2
    return Ag