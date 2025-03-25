import re


def bitstuffX(BinaryText):
    pattern = "101"
    result = []
    for text in BinaryText:
        result.append(re.sub(pattern, pattern + "0", text))
    return result


def bitstuffY(BinaryText):
    pattern = "01"
    result = []
    for text in BinaryText:
        result.append(re.sub(pattern, pattern + "0", text))
    return result

def bitstuffZ(BinaryText):
    pattern = "1"
    result = []
    for text in BinaryText:
        result.append(re.sub(pattern, pattern + "0", text))
    return result
