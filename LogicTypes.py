import operator

# CONSTANTS
TRUTH_TABLE_PATH = "E:\\ise_masters\\gal_thesis\\data_sets\\circuits\\truth_tables\\"
CIRCUIT_SYSTEM_PATH = "E:\\ise_masters\\gal_thesis\\data_sets\\circuits\\Data_Systems\\"


class GateFeature:
    def __init__(self, gate, inputs):
        self.gate = gate
        self.inputs = []
        for inp in inputs:
            self.inputs.append(inp if isinstance(inp, str) else inp.copy())
        self.outputs = set()

    def to_string(self):
        to_string = ""
        if self.gate is not None:
            to_string += str(self.gate.name) + "("
            for input_index in range(len(self.inputs)):
                if input_index > 0: # add comma for all inputs but the first input
                    to_string += ","
                if isinstance(self.inputs[input_index], str): # only argument
                    to_string += str(self.inputs[input_index])
                else:
                    to_string += str(self.inputs[input_index].to_string())
            to_string += ")"
        else: # only argument
            return self.inputs[0]

        return to_string

    def __key(self):
        # return (self.attr_a, self.attr_b, self.attr_c)
        return self.to_string()

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        return isinstance(self, type(other)) and self.__key() == other.__key()

    def copy(self):
        return GateFeature(self.gate, [inp if isinstance(inp, str) else inp.copy() for inp in self.inputs])


class OneNot:
    name = "inverter"
    numInputs = 1
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value):
        return not value


class TwoXor:
    name = "xor2"
    numInputs = 2
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value1, value2):
        return operator.xor(value1, value2)


class TwoAnd:
    name = "and2"
    numInputs = 2
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value1, value2):
        return operator.and_(value1, value2)


class ThreeAnd:
    name = "and3"
    numInputs = 3
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value1, value2, value3):
        return operator.and_(operator.and_(value1, value2), value3)


class FourAnd:
    name = "and4"
    numInputs = 4
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value1, value2, value3, value4):
        return operator.and_(operator.and_(value1, value2), operator.and_(value3, value4))


class FiveAnd:
    name = "and5"
    numInputs = 5
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value1, value2, value3, value4, value5):
        return operator.and_(operator.and_(operator.and_(value1, value2), operator.and_(value3, value4)), value5)


class NotAnd:
    name = "nand2"
    numInputs = 2
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value1, value2):
        return OneNot.f(operator.and_(value1, value2))


class TwoOr:
    name = "or2"
    numInputs = 2
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value1, value2):
        return operator.or_(value1, value2)


class ThreeOr:
    name = "or3"
    numInputs = 3
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value1, value2, value3):
        return operator.or_(operator.or_(value1, value2), value3)


class FourOr:
    name = "or4"
    numInputs = 4
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value1, value2, value3, value4):
        return operator.or_(operator.or_(value1, value2), operator.or_(value3, value4))


class TwoNor:
    name = "nor2"
    numInputs = 2
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value1, value2):
        return not operator.or_(value1, value2)


class ThreeNor:
    name = "nor3"
    numInputs = 3
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value1, value2, value3):
        return not operator.or_(operator.or_(value1, value2), value3)


class FourNor:
    name = "nor4"
    numInputs = 4
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value1, value2, value3, value4):
        return not operator.or_(operator.or_(value1, value2), operator.or_(value3, value4))


class FiveNor:
    name = "nor5"
    numInputs = 5
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value1, value2, value3, value4, value5):
        return not operator.or_(operator.or_(operator.or_(value1, value2), operator.or_(value3, value4)), value5)


class Buffer:
    name = "buffer"
    numInputs = 1
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value1):
        return value1