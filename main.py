from PIL import Image
import heapq
from collections import Counter
import numpy as np
import wave
from pydub import AudioSegment


### Utility Functions ###


def read_audio(file_path, audio_format):
    if audio_format == "wav":
        return read_wav(file_path)
    elif audio_format == "mp3":
        return read_mp3(file_path)
    return None


def read_wav(file_path):
    with wave.open(file_path, "r") as wav_file:
        n_channels, _, framerate, n_frames = wav_file.getparams()[:4]
        audio_data = wav_file.readframes(n_frames)
        audio_data = np.frombuffer(audio_data, dtype=np.int16)
        audio_data = audio_data.reshape((-1, n_channels))
        return audio_data.flatten()


def read_mp3(file_path):
    audio = AudioSegment.from_mp3(file_path)
    audio_data = np.array(audio.get_array_of_samples())
    return audio_data.flatten()


def read_data(file_path, data_type):
    if data_type == "image":
        return read_image(file_path)
    elif data_type == "audio":
        return read_audio(file_path, file_path.split(".")[-1])
    elif data_type == "text":
        return read_text(file_path)
    return None


def read_image(file_path):
    img = Image.open(file_path).convert("L")
    return np.array(img)


def read_text(file_path):
    with open(file_path, "rb") as f:
        return f.read()


### Compression Algorithms ###
def run_length_encoding(data):
    n = len(data)
    i = 0
    compressed_data = ""

    while i < n - 1:
        count = 1
        while i < n - 1 and data[i] == data[i + 1]:
            count += 1
            i += 1
        i += 1
        compressed_data += f"{data[i - 1]}{count}"

    return compressed_data


def arithmetic_coding(data):
    radix = 10
    enc, pow, freq = arithmethic_coding_helper(data, radix)
    return f"{enc} * {radix}^{pow}"


def huffman_coding(data):
    compressed_data = huffman_encoding(data)
    return compressed_data


def dictionary_based_coding(data):
    compressed_data = dictionary_based_coding_helper(data)
    return compressed_data


### Huffman ###


# Huffman tree node
class Node:
    def __init__(self, char, freq):
        self.char = char  # Character stored in the node
        self.freq = freq  # Frequency of the character
        self.left = None  # Left child
        self.right = None  # Right child

    # For heap operations we define a less than ('<') operator
    # The node with lower frequency will be considered 'smaller'
    def __lt__(self, other):
        return self.freq < other.freq


# Generate Huffman codes
def generate_huffman_codes(root, code, huffman_codes):
    if root is None:  # Base case: If root is None, return
        return

    # If it's a leaf node, store the Huffman code
    if root.left is None and root.right is None:
        huffman_codes[root.char] = code

    # Recursive calls for left and right subtrees
    generate_huffman_codes(root.left, code + "0", huffman_codes)
    generate_huffman_codes(root.right, code + "1", huffman_codes)


# Huffman encoding
def huffman_encoding(s):
    # Count frequency of each character
    frequency = Counter(s)

    # Initialize priority queue
    priority_queue = [Node(char, freq) for char, freq in frequency.items()]
    heapq.heapify(priority_queue)

    # Build the Tree
    while len(priority_queue) > 1:
        left = heapq.heappop(priority_queue)
        right = heapq.heappop(priority_queue)

        merged = Node(None, left.freq + right.freq)
        merged.left = left
        merged.right = right

        heapq.heappush(priority_queue, merged)

    root = heapq.heappop(priority_queue)

    huffman_codes = {}
    generate_huffman_codes(root, "", huffman_codes)

    compressed_data = "".join([huffman_codes[char] for char in s])

    return compressed_data


### Arithmetic ###
# Function to calculate cumulative frequency for Arithmetic coding
def cumulative_freq(freq):
    cf = {}
    total = 0
    for b in sorted(freq.keys()):
        cf[b] = total
        total += freq[b]
    return cf


# Function for Arithmetic coding
def arithmethic_coding_helper(bytes, radix=10):
    freq = Counter(bytes)
    cf = cumulative_freq(freq)
    base = len(bytes)
    lower = 0
    pf = 1

    for b in bytes:
        # print(f"b: {b}, lower: {lower}, pf: {pf}")  # Debugging
        b = int(b) if isinstance(b, (int, np.int64)) else ord(b)  # Updated line
        lower = lower * base + cf[b] * pf
        pf *= freq[b]

    upper = lower + pf

    pow = 0
    while True:
        pf //= radix
        if pf == 0:
            break
        print(pow)
        pow += 1

    enc = (upper - 1) // radix**pow
    return enc, pow, freq


# LZW Algorithm
def compress(uncompressed):
    """Compress a string to a list of output symbols."""

    dict_size = 256
    dictionary = dict((chr(i), i) for i in range(dict_size))

    w = ""
    result = []
    for c in uncompressed:
        wc = w + c
        if wc in dictionary:
            w = wc
        else:
            result.append(dictionary[w])
            dictionary[wc] = dict_size
            dict_size += 1
            w = c

    if w:
        result.append(dictionary[w])
    return result


# Function to implement Dictionary-based (LZW) compression
def dictionary_based_coding_helper(data):
    if isinstance(data, np.ndarray):
        flat_data = data.flatten()
        uncompressed_str = "".join(map(chr, flat_data))
    else:
        uncompressed_str = "".join(map(chr, data))

    compressed_data = compress(uncompressed_str)

    return compressed_data


# Main function
def main():
    file_path = input("Enter file path: ")
    data_type = input("Enter data type (image/audio/text/): ")
    algorithm = input(
        "Enter compression algorithm (RLE/Arithmetic/Huffman/Dictionary): "
    )

    data = read_data(file_path, data_type)

    if data is None:
        print("Failed to read data.")
        return

    if data_type == "image":
        flat_array = data.flatten()

        if algorithm == "RLE":
            compressed_data = run_length_encoding(flat_array)
        elif algorithm == "Arithmetic":
            flat_array = data.flatten().astype(int)

            if data_type in [
                "image",
                "text",
            ]:
                compressed_data = arithmetic_coding(flat_array)  # Use flattened array
            else:
                print("arithmetic coding is only allowed for image and text.")
                return
        elif algorithm == "Huffman":
            compressed_data = huffman_coding(flat_array)
        else:
            if data_type in [
                "image",
                "text",
            ]:
                compressed_data = dictionary_based_coding(flat_array)
            else:
                print("Dictionary-based coding is only allowed for image and text.")
                return
    else:
        if algorithm == "RLE":
            compressed_data = run_length_encoding(data)
        elif algorithm == "Arithmetic":
            if data_type in [
                "image",
                "text",
            ]:
                compressed_data = arithmetic_coding(data)  # Use flattened array
            else:
                print("Arithmetic coding is only allowed for image and text.")
                return
        elif algorithm == "Huffman":
            compressed_data = huffman_coding(data)
        else:
            if data_type in [
                "image",
                "text",
            ]:
                compressed_data = dictionary_based_coding(data)
            else:
                print("Dictionary-based coding is only allowed for image and text.")
                return

    print("Compressed Data:", compressed_data)


if __name__ == "__main__":
    main()
