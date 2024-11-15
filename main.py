import numpy as np
import random

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

def apply_noise(bits, probability):
    flip = {'0': '1', '1' : '0'}
    bits_after_noise = ''
    for bit in bits:
        bits_after_noise += flip[bit] if random.random() < probability else bit
        
    return bits_after_noise

   
charcodes = get_charcodes('files/charactcodif.txt')
coded = letters_to_bits("cada", charcodes)
print(coded)

print(apply_noise(coded, 0.04))




