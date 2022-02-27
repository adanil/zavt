class BitInputStream:

    def __init__(self, inp):
        self.input = inp
        self.currentbyte = 0
        self.numbitsremaining = 0


    def read(self):
        if self.currentbyte == -1:
            return -1
        if self.numbitsremaining == 0:
            temp = self.input.read(1)
            if len(temp) == 0:
                self.currentbyte = -1
                return -1
            self.currentbyte = temp[0]
            self.numbitsremaining = 8
        assert self.numbitsremaining > 0
        self.numbitsremaining -= 1
        return (self.currentbyte >> self.numbitsremaining) & 1


    def read_no_eof(self):
        result = self.read()
        if result != -1:
            return result
        else:
            raise EOFError()

    def read_code_bit(self):
        temp = self.read()
        if temp == -1:
            temp = 0
        return temp

    def close(self):
        self.input.close()
        self.currentbyte = -1
        self.numbitsremaining = 0



class BitOutputStream:

    def __init__(self, out):
        self.output = out
        self.currentbyte = 0
        self.numbitsfilled = 0

    def write(self, b):
        if b not in (0, 1):
            raise ValueError("Argument must be 0 or 1")
        self.currentbyte = (self.currentbyte << 1) | b
        self.numbitsfilled += 1
        if self.numbitsfilled == 8:
            towrite = bytes((self.currentbyte,))
            self.output.write(towrite)
            self.currentbyte = 0
            self.numbitsfilled = 0


    def close(self):
        while self.numbitsfilled != 0:
            self.write(0)
