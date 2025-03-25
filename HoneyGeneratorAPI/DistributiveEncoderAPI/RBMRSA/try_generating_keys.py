import Crypto.Util.number
import math
import sys
import random as random


def compute_bit(bit_input):
    bits = math.floor(bit_input / 4)
    # Commented this part, as it conflicts when running encrypt.py with arguments.
    # if len(sys.argv) > 1:
    #     bits = int(sys.argv[1])

    return bits


def generating_keys(bits):
    p = Crypto.Util.number.getPrime(bits, randfunc=Crypto.Random.get_random_bytes)

    q = Crypto.Util.number.getPrime(bits, randfunc=Crypto.Random.get_random_bytes)
    while p == q:
        q = Crypto.Util.number.getPrime(bits, randfunc=Crypto.Random.get_random_bytes)

    r = Crypto.Util.number.getPrime(bits, randfunc=Crypto.Random.get_random_bytes)
    while r == p or r == q:
        r = Crypto.Util.number.getPrime(bits, randfunc=Crypto.Random.get_random_bytes)

    s = Crypto.Util.number.getPrime(bits, randfunc=Crypto.Random.get_random_bytes)
    while s == p or s == q or s == r:
        s = Crypto.Util.number.getPrime(bits, randfunc=Crypto.Random.get_random_bytes)

    return p, q, r, s

def gen_e(PHI):
    e = random.randint(2, 1000000)
    while math.gcd (e, PHI) != 1:
        e = random.randint(2, 1000000)
    return e

def computation_keys(p, q, r, s):
    N = p * q * r * s
    PHI = (p - 1) * (q - 1) * (r - 1) * (s - 1)
    #e = 65537
    e = gen_e(PHI)
    #d = Crypto.Util.number.inverse(e, PHI)

    return N, PHI, e

