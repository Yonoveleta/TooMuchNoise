import numpy as np
import random
import itertools
from numpy.linalg import norm
import re

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

def segment_bits(bits, segment_size):
    return [bits[bit:bit+segment_size] for bit in range(0, len(bits), segment_size)]
    

def get_charcodes(filename):
    charcodes = {}

    with open(filename, 'r') as file:
        for line in file:
            charcodes[line[0]] = line[1:-1]
            
    return charcodes

def chars_to_bits(msg, charcodes):
    code = ''
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
    segment_size = len(charcodes[next(iter(charcodes))])
    segmented_bits = segment_bits(bits, segment_size)

    charcodes_chars = list(charcodes.keys())
    charcodes_bits = list(charcodes.values())
    
    original_chars = ''
    
    for current_bits in segmented_bits:
        stringified_bits = re.sub(r'[\[\]\s]', '', np.array2string(current_bits))
        original_chars += charcodes_chars[charcodes_bits.index(stringified_bits)]
        
    return original_chars

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

def get_most_similar_segment(segment, possible_segments):
    cosine_similarity = 0
    most_similar_segment = None
    
    for possible_segment in possible_segments[1:]:
        cosine = np.dot(segment, possible_segment)/(norm(segment)*norm(possible_segment))
        if cosine > cosine_similarity:
            cosine_similarity = cosine
            most_similar_segment = possible_segment
    
    return most_similar_segment

def retrieve_information(received_bits, redundancy_matrix):
    segment_size = len(redundancy_matrix[0])
    segmented_bits = segment_bits(received_bits, segment_size)
    
    original_segment_size = len(redundancy_matrix)
    possible_original_segments = get_nbits_combinations(original_segment_size)
    possible_segments = [np.matmul(possible_original_segments[segment], redundancy_matrix) % 2 for segment in range(len(possible_original_segments))]
    
    non_redundant_bits = np.array([], dtype=np.uint8)
    for segment in segmented_bits:
        non_redundant_bits = np.concatenate((non_redundant_bits, get_most_similar_segment(segment, possible_segments)[:original_segment_size]))
        
    return non_redundant_bits
    
        
charcodes = get_charcodes('files/charactcodif.txt')

bits = chars_to_bits("cada", charcodes)

bits_with_redundancy = add_redundancy(bits, H_4_7)

bits_after_noise = apply_noise(bits_with_redundancy, 0.04)

retrieved_bits = retrieve_information(bits_after_noise, H_4_7)

original_chars = bits_to_chars(retrieved_bits, charcodes)

print(original_chars)