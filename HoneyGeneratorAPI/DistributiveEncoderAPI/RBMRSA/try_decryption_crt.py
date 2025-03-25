import math

def crt_equations (p, q , r, s, N, d):
    dp = d % (p -1)
    dq = d % (q - 1)
    dr = d % (r - 1)
    ds = d % (s - 1)

    return dp, dq, dr, ds

def modinv (a, m):
    """Calculate the modular inverse of a modulo m."""
    m0, x0, x1 = m, 0, 1
    while a > 1:
        q = a // m
        m, a = a % m, m
        x0, x1 = x1 - q * x0, x0
    return x1 + m0 if x1 < 0 else x1


# Calculate modular inverses
def modInv_Computation (N, p, q, r, s):
    pInv = modinv(N // p, p)
    qInv = modinv(N // q, q)
    rInv = modinv(N // r, r)
    sInv = modinv(N // s, s)

    return pInv, qInv, rInv, sInv
    

def four_parts (CipherText, p, q, r, s, N, pInv, qInv, rInv, sInv, dp, dq, dr, ds):
    message = []
    for text in CipherText:
        M1 = pow(text, dp, p) * q * r * s * pInv 
        M2 = pow(text, dq, q) * p * r * s * qInv 
        M3 = pow(text, dr, r) * p * q * s * rInv 
        M4 = pow(text, ds, s) * p * q * r * sInv
        final_CT = (M1 + M2 + M3 + M4) % N
        message.append(final_CT)
    return message

if __name__ == "__main__": # ----- for checking only!!! 
    N = 2145
    p = 3
    q = 5
    r = 11
    s = 13
    d = 859
    CipherText = 1667

    pInv, qInv, rInv, sInv = modInv_Computation(N, p, q, r, s)
    dp, dq, dr, ds = crt_equations(p, q , r, s, N, d)
    DecryptedText = four_parts (CipherText, p, q, r, s, N, pInv, qInv, rInv, sInv, dp, dq, dr, ds)

    print (dp, dq, dr, ds)
    print (pInv, qInv, rInv, sInv)
    print (DecryptedText)


