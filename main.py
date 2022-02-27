
import contextlib, sys
import ArCoding as arithmeticcoding
import os
import Model
import numpy as np
import time

import ArithmeticCoder as myCoder
import BitStreams

class modelProbabilities:
    def __init__(self,alphabet_size = 256,freqs = None):
        self.alphabet_size = alphabet_size
        self.total = 0
        self.cumul = []
        self.freqs = [0 for i in range(alphabet_size)]

    def get_probs(self,probs):
        self.total = 10000
        fr_sum = 0
        max_ind = -1
        max_value = 0
        for i in range(self.alphabet_size):
            self.freqs[i] = int(probs[i] * self.total)
            if self.freqs[i] > max_value:
                max_value = self.freqs[i]
                max_ind = i
            fr_sum += self.freqs[i]
        for i in range(self.alphabet_size):
            if self.freqs[i] == 0:
                if fr_sum < self.total:
                    self.freqs[i] = 1
                    fr_sum += 1
                else:
                    self.freqs[i] = 1
                    self.freqs[max_ind] -= 1

        self.freqs = np.array(self.freqs)

    def calc_cumulative(self):
        self.cumul.clear()
        self.cumul.append(0)
        curr = 0
        for i in self.freqs:
            curr += i
            self.cumul.append(curr)

    def get_low(self,symbol):
        return self.cumul[symbol]

    def get_high(self,symbol):
        return self.cumul[symbol + 1]

    def find_symbol_by_cum_value(self,cum_value):
        answer = -1
        start = 0
        end = len(self.cumul)

        while start < end:
            middle = start + (end - start) // 2

            if cum_value >= self.cumul[middle]:
                answer = middle
                start = middle + 1
            else:
                end = middle

        return answer

    def get_total(self):
        return self.total


def split_file(file_path,part_size_bytes):
    part_numb = 1
    out_name = 'test_parts/part' + str(part_numb)
    readBytes = 0
    out_file = open(out_name,'wb')
    with open(file_path,'rb') as inp:
        while True:
            b = inp.read(1)
            if (len(b) == 0):
                break
            out_file.write(b)
            readBytes += 1
            if (readBytes % part_size_bytes == 0):
                out_file.close()
                part_numb += 1
                out_name = 'test_parts/part' + str(part_numb)
                out_file = open(out_name,'wb')
    out_file.close()


def encode_file(input_file,output_file,model_conf):
    filesize = os.path.getsize(input_file)
    read_bytes = 0
    alphabet_size = 256

    init_probs = [1 / alphabet_size for i in range(alphabet_size)]
    mp = modelProbabilities()
    mp.get_probs(init_probs)
    mp.calc_cumulative()

    net = Model.load_model(model_conf)
    print(net)
    net.eval()
    h = net.init_hidden(1)

    outf = open(output_file, 'wb')
    outf.write((filesize).to_bytes(8, byteorder='big', signed=False))

    outstream = BitStreams.BitOutputStream(outf)
    coder = myCoder.ArithmeticEncoder(32, outstream)
    with open(input_file, 'rb') as inp:
        while True:
            symbol = inp.read(1)
            read_bytes += 1
            symbol = ord(symbol)
            coder.process(symbol, mp)
            if read_bytes == filesize:
                break

            probs, h = Model.predict(net, symbol, h)
            probs = probs.flatten().cpu().numpy()
            mp.get_probs(probs)
            mp.calc_cumulative()
            if read_bytes % 1000 == 0:
                print(read_bytes, "/", filesize)
        coder.finish()

    outstream.close()
    outf.close()

    return os.path.getsize(output_file)


def decode_file(encoded_file,decoded_file,model_conf):
    inf = open(encoded_file, 'rb')
    encoded_filesize = int.from_bytes(inf.read(8), byteorder='big')
    print('Filesize to decode: ', encoded_filesize)
    instream = BitStreams.BitInputStream(inf)

    alphabet_size = 256

    d_net = Model.load_model(model_conf)
    print(d_net)
    d_net.eval()
    h = d_net.init_hidden(1)

    init_probs = [1 / alphabet_size for i in range(alphabet_size)]
    mp = modelProbabilities()
    mp.get_probs(init_probs)
    mp.calc_cumulative()

    count_decoded_symbols = 0

    decoder = myCoder.ArithmeticDecoder(32, instream)
    with open(decoded_file, 'wb') as dec_out:
        while True:
            symb = decoder.process(mp)
            count_decoded_symbols += 1
            dec_out.write(bytes((symb,)))
            if count_decoded_symbols % 1000 == 0:
                print(count_decoded_symbols, "/", encoded_filesize)
            if count_decoded_symbols == encoded_filesize:
                break
            probs, h = Model.predict(d_net, symb, h)
            probs = probs.flatten().cpu().numpy()
            mp.get_probs(probs)
            mp.calc_cumulative()

    instream.close()
    inf.close()

    return os.path.getsize(decoded_file)

def compare_files(file1,file2):
    f1 = open(file1, 'rb')
    content1 = f1.read()
    f2 = open(file2,'rb')
    content2 = f2.read()
    f1.close()
    f2.close()
    equal = True
    if len(content1) != len(content2):
        equal = False
        print(len(content1), ' : ' ,len(content2))
    if (equal):
        for i in range(len(content1)):
            if content1[i] != content2[i]:
                equal = False
                print(i,': ',content1[i],' ',content2[i])

    return equal

def myMain():
    input_file = "test_parts/part1"
    output_file = "output/encoded.zavt"
    decoded_file = "output/out"
    model_conf = "models/lstmv1_50_epoch.net"

    print("Input file: ",input_file)
    filesize_to_encode = os.path.getsize(input_file)
    print("Input filesize: ",filesize_to_encode)
    print("Encode...")
    start_enc = time.time()
    comp_size = encode_file(input_file,output_file,model_conf)
    end_enc = time.time()
    encode_time = end_enc - start_enc
    print("Decode...")
    start_dec = time.time()
    decomp_size = decode_file(output_file,decoded_file,model_conf)
    end_dec = time.time()
    decode_time = end_dec - end_enc
    if compare_files(input_file,decoded_file):
        print("Input file equals decoded file!")
    else:
        print('Incorrect compression :(')
    comp_ratio = filesize_to_encode/comp_size
    print(filesize_to_encode,' ------> ',comp_size)
    print('CR: ',comp_ratio)
    size_in_MB = filesize_to_encode/(1024*1024)
    print("Encode speed : ", (size_in_MB)/encode_time,"Mb/s Decode speed: ",size_in_MB/decode_time,"Mb/s")


if __name__ == "__main__":
    myMain()
    # split_file("/Users/daniilavtusko/Desktop/Study/ВКР/text8",10000)