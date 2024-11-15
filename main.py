import numpy as np

def get_charcodes(filename):
    charcodes = {}

    with open(filename, 'r') as file:
        for line in file:
            charcodes[line[0]] = line[1:-1]
            
    return charcodes


def letters_to_bits(msg, charcodes):
    code = ""
    for letter in msg:
        code += charcodes[letter]
    return code

charcodes = get_charcodes('files/charactcodif.txt')
print(letters_to_bits("cada", charcodes))

