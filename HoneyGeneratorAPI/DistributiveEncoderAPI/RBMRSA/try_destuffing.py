import re


def destuffZ(BinaryText):
    patterns = "10"
    destuff = []
    for text in BinaryText: 
        destuff.append (re.sub(patterns, "1", text))
    return destuff


def destuffY(BinaryText):
    patterns = "010"
    destuff = []
    for text in BinaryText: 
        destuff.append (re.sub(patterns, "01", text))
    return destuff


def destuffX(BinaryText):
    patterns = "1010"
    destuff = []
    for text in BinaryText: 
        destuff.append (re.sub(patterns, "101", text))
    return destuff



