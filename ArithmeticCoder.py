class ArithmeticEncoder:
    def __init__(self,numbits,outstream):
        self.numbits = numbits
        self.range = 1 << numbits
        self.low = 0
        self.high = self.range - 1
        self.half_range = self.range >> 1
        self.q1_range = self.half_range >> 1
        self.state_mask = self.range - 1

        self.outputStream = outstream
        self.follow_bits = 0

    def output(self):
        bit = self.low >> (self.numbits - 1)
        self.outputStream.write(bit)
        for _ in range(self.follow_bits):
            self.outputStream.write(bit ^ 1)
        self.follow_bits = 0

    def finish(self):
        self.outputStream.write(1)


    def process(self,symbol,freqs):
        range = self.high - self.low + 1
        total = freqs.get_total()
        symlow = freqs.get_low(symbol)
        symhigh = freqs.get_high(symbol)

        # print("Low: ",self.low, " high: ", self.high)
        low = self.low
        self.low = low + symlow*range // total
        self.high = low + symhigh*range // total - 1

        while ((self.low ^ self.high) & self.half_range) == 0:
            self.output()
            self.low = ((self.low << 1) & self.state_mask)
            self.high = ((self.high << 1) & self.state_mask) | 1

        while (self.low & ~self.high & self.q1_range) != 0:
            self.follow_bits += 1
            self.low = (self.low << 1) ^ self.half_range
            self.high = ((self.high ^ self.half_range) << 1) | self.half_range | 1

class ArithmeticDecoder:
    def __init__(self,numbits,instream):
        self.numbits = numbits
        self.range = 1 << numbits
        self.low = 0
        self.high = self.range - 1
        self.half_range = self.range >> 1
        self.q1_range = self.half_range >> 1
        self.state_mask = self.range - 1

        self.instream = instream
        self.follow_bits = 0

        self.code = 0

        for _ in range(self.numbits):
            self.code = (self.code << 1) | self.instream.read_code_bit()

    def update_code(self):
        self.code = ((self.code << 1) & self.state_mask) | self.instream.read_code_bit()

    def update_underflow(self):
        self.code = (self.code & self.half_range) | ((self.code << 1) & (self.state_mask >> 1)) | self.instream.read_code_bit()

    def process(self,freqs):
        total = freqs.get_total()
        range = self.high - self.low + 1
        offset = self.code - self.low
        cum_value = ((offset + 1)*total - 1) // range
        # print("Cumul: ",cum_value)

        # if (cum_value >= total - 1):
        #     print("EOF")

        symbol = freqs.find_symbol_by_cum_value(cum_value)
        # print(chr(symbol))
        # print("Low: ",self.low, " High: ", self.high, " Code: ",self.code)

        symlow = freqs.get_low(symbol)
        symhigh = freqs.get_high(symbol)

        # print("Low: ", self.low, " high: ", self.high)
        low = self.low
        self.low = low + symlow * range // total
        self.high = low + symhigh * range // total - 1

        while ((self.low ^ self.high) & self.half_range) == 0:
            self.update_code()
            self.low = ((self.low << 1) & self.state_mask)
            self.high = ((self.high << 1) & self.state_mask) | 1

        while (self.low & ~self.high & self.q1_range) != 0:
            self.update_underflow()
            self.low = (self.low << 1) ^ self.half_range
            self.high = ((self.high ^ self.half_range) << 1) | self.half_range | 1

        return symbol




