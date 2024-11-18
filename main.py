import numpy as np
import random
import itertools

H_4_7 = np.array([
    [1, 0, 0, 0, 1, 1, 0],
    [0, 1, 0, 0, 1, 0, 1],
    [0, 0, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1],
], dtype=np.uint8)

G_12_23 = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
], dtype=np.uint8)

def get_charcodes(filename):
    charcodes = {}

    with open(filename, 'r') as file:
        for line in file:
            charcodes[line[0]] = line[1:-1]
            
    return charcodes

def chars_to_bits(msg, charcodes):
    code = ""
    for letter in msg:
        code += charcodes[letter]
    return code

def apply_noise(bits, probability):
    flip = {0:1, 1:0}
    bits_after_noise = np.array([], dtype=np.uint8)
    for bit in bits:
        bits_after_noise = np.append(bits_after_noise, flip[bit] if random.random() < probability else bit)
        
    return bits_after_noise

def bits_to_array(bits):
    return np.array([np.uint8(bit) for bit in bits], dtype=np.uint8)

def bits_to_chars(bits, charcodes):
    bit_size = len(charcodes[next(iter(charcodes))])
    splitted_bits = [bits[bit:bit+bit_size] for bit in range(0, len(bits), bit_size)]

    charcodes_chars = list(charcodes.keys())
    charcodes_bits = list(charcodes.values())

    return ''.join([charcodes_chars[charcodes_bits.index(bits)] for bits in splitted_bits])

def add_redundancy(bits, redundancy_matrix):
    segment_size = len(redundancy_matrix)
    segmented_bits = [bits[bit:bit+segment_size] for bit in range(0, len(bits), segment_size)]
    bits_with_redundancy = np.array([], dtype=np.uint8)
    for segment in segmented_bits:
        splitted_segment = bits_to_array(segment)
        bits_with_redundancy = np.concatenate((bits_with_redundancy, np.matmul(splitted_segment, redundancy_matrix) % 2))
    
    return bits_with_redundancy

def get_nbits_combinations(n):
    return list(itertools.product([0,1], repeat=n))
        
charcodes = get_charcodes('files/charactcodif.txt')

bits = chars_to_bits("cada", charcodes)
print(bits)

bits_with_redundancy = add_redundancy(bits, H_4_7)
print(bits_with_redundancy)

bits_after_noise = apply_noise(bits_with_redundancy, 0.04)
print(bits_after_noise)

print(get_nbits_combinations(4))