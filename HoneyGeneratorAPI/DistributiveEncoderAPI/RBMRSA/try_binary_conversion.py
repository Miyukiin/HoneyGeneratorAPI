import math

def decimal_to_binary(CipherText):
    BinaryText = []
    for decimal_number in CipherText:
        binary_string = ""
        while decimal_number > 0:
            remainder = decimal_number % 2
            binary_string = str(remainder) + binary_string
            decimal_number //= 2
        BinaryText.append(binary_string)
    return BinaryText


def binary_to_decimal(BinaryText):
    decimal = 0
    for i, digit in enumerate(BinaryText[::-1]):
        if digit == '1':
            decimal += 2 ** i
    return decimal