import operator

# CONSTANTS
TRUTH_TABLE_PATH = "E:\\ise_masters\\gal_thesis\\data_sets\\circuits\\truth_tables\\"
CIRCUIT_SYSTEM_PATH = "E:\\ise_masters\\gal_thesis\\data_sets\\circuits\\Data_Systems\\"
ORIGINAL_CIRCUITS_TO_STRING_PATH = "E:\\ise_masters\\gal_thesis\\data_sets\\circuits\\original_circuits_to_string\\"

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

    def deep_equals(self, other):
        # what about basic inputs??

        if not isinstance(self, type(other)):
            return False
        if len(self.inputs) != len(other.inputs):
            return False
        if self.gate is None:  #basic inputs
            if other.gate is None:
                if self.inputs[0] != other.inputs[0]:
                    return False
                else:
                    return True
            else:
                return False

        equal = True
        for inp in self.inputs:
            if not equal:
                return False
            else:
                for other_inp in other.inputs:
                    if inp.deep_equals(other_inp):
                        equal = True
                        break
                    equal = False

        return equal


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


class EightAnd:
    name = "and8"
    numInputs = 8
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value1, value2, value3, value4, value5, value6, value7, value8):
        return operator.and_(operator.and_(operator.and_(value1, value2), operator.and_(value3, value4)),
                                           operator.and_(value5, value6), operator.and_(value7, value8))


class NineAnd:
    name = "and9"
    numInputs = 9
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value1, value2, value3, value4, value5, value6, value7, value8, value9):
        return operator.and_(operator.and_(operator.and_(operator.and_(value1, value2), operator.and_(value3, value4)),
                                           operator.and_(value5, value6), operator.and_(value7, value8)), value9)


class NotAnd:
    name = "nand2"
    numInputs = 2
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value1, value2):
        return OneNot.f(operator.and_(value1, value2))


class ThreeNand:
    name = "nand3"
    numInputs = 4
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value1, value2, value3):
        return OneNot.f(operator.and_(operator.and_(value1, value2), value3))


class FourNand:
    name = "nand4"
    numInputs = 4
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value1, value2, value3, value4):
        return OneNot.f(operator.and_(operator.and_(value1, value2), operator.and_(value3, value4)))


class FiveNand:
    name = "nand5"
    numInputs = 5
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value1, value2, value3, value4, value5):
        return OneNot.f(operator.and_(operator.and_(operator.and_(value1, value2), operator.and_(value3, value4)), value5))


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


class SixMux:
    name = "mux6"
    numInputs = 6
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value1, value2, value3, value4, value5, value6):
        return FourOr.f(ThreeAnd.f(OneNot.f(value1), OneNot.f(value2), value3), ThreeAnd.f(value1, OneNot.f(value2), value4),
                        ThreeAnd.f(OneNot.f(value1), value2, value5), ThreeAnd.f(value1, value2, value6))

class mux3:
    name = "mux3"
    numInputs = 5
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value1, value2, value3, value4, value5):
        return ThreeOr.f(ThreeAnd.f(value1, OneNot.f(value4), OneNot.f(value5)),
                       ThreeAnd.f(value2, OneNot.f(value4), value5),
                       ThreeAnd.f(value3, value4, OneNot.f(value5)))


class adder1_o1:
    name = "adder1_o1"
    numInputs = 3
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value1, value2, value3):
        return TwoOr.f(TwoAnd.f(value1, value2),
                       TwoAnd.f(value3, TwoXor.f(value1, value2)))


class adder1_o2:
    name = "adder1_o2"
    numInputs = 3
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value1, value2, value3):
        return TwoXor.f(value3, TwoXor.f(value1, value2))


class adder2_o1:
    name = "adder2_o1"
    numInputs = 5
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value1, value2, value3, value4, value5):
        return TwoXor.f(value5, TwoXor.f(value1,value2))


class adder2_o2:
    name = "adder2_o2"
    numInputs = 5
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value1, value2, value3, value4, value5):
        return TwoXor.f(TwoOr.f(TwoAnd.f(value1, value2), TwoAnd.f(value5, TwoXor.f(value1,value2))),
                        TwoXor.f(value3, value4)
                        )


class adder2_o3:
    name = "adder2_o3"
    numInputs = 5
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value1, value2, value3, value4, value5):
        return TwoOr.f(TwoAnd.f(value3, value4),
                       TwoAnd.f(TwoOr.f(TwoAnd.f(value1, value2), TwoAnd.f(value5, TwoXor.f(value1,value2))),
                                TwoXor.f(value3, value4)
                                )
                       )


class barrel_shifter3_o1:
    name = "barrel_shifter3_o1"
    numInputs = 5
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value1, value2, value3, value4, value5):
        z1 = OneNot.f(value1)
        z2 = OneNot.f(value2)
        m1_1_q1 = TwoAnd.f(value3, z1)
        m1_1_q2 = TwoAnd.f(value4, value1)
        z1_1 = TwoOr.f(m1_1_q1, m1_1_q2)
        z1_3 = TwoAnd.f(value5, z1)
        m2_1_q1 = TwoAnd.f(z1_1, z2)
        m2_1_q2 = TwoAnd.f(z1_3, value2)

        return TwoOr.f(m2_1_q1, m2_1_q2)


class barrel_shifter3_o2:
    name = "barrel_shifter3_o2"
    numInputs = 5
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value1, value2, value3, value4, value5):
        z1 = OneNot.f(value1)
        z2 = OneNot.f(value2)
        m1_2_q1 = TwoAnd.f(value4, z1)
        m1_2_q2 = TwoAnd.f(value5, value1)
        z1_2 = TwoOr.f(m1_2_q1, m1_2_q2)

        return TwoAnd.f(z1_2, z2)


class barrel_shifter3_o3:
    name = "barrel_shifter3_o3"
    numInputs = 5
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value1, value2, value3, value4, value5):
        z1 = OneNot.f(value1)
        z2 = OneNot.f(value2)
        z1_3 = TwoAnd.f(value5, z1)

        return TwoAnd.f(z1_3, z2)


class comp2_o1:
    name = "comp2_o1"
    numInputs = 4
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value1, value2, value3, value4):
        eq1 = OneNot.f(TwoXor.f(value1, value2))
        eq2 = OneNot.f(TwoXor.f(value3, value4))
        o2 = TwoAnd.f(eq1, eq2)
        b2_n = OneNot.f(value2)
        b1_n = OneNot.f(value4)
        g1 = TwoAnd.f(value1, b2_n)
        g2 = ThreeAnd.f(eq1, value3, b1_n)
        o3 = TwoOr.f(g1, g2)
        eq_gt = TwoOr.f(o3, o2)
        o1 = OneNot.f(eq_gt)
        return o1


class comp2_o2:
    name = "comp2_o2"
    numInputs = 4
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value1, value2, value3, value4):
        eq1 = OneNot.f(TwoXor.f(value1, value2))
        eq2 = OneNot.f(TwoXor.f(value3, value4))
        o2 = TwoAnd.f(eq1, eq2)
        return o2


class comp2_o3:
    name = "comp2_o3"
    numInputs = 4
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value1, value2, value3, value4):
        eq1 = OneNot.f(TwoXor.f(value1, value2))
        b2_n = OneNot.f(value2)
        b1_n = OneNot.f(value4)
        g1 = TwoAnd.f(value1, b2_n)
        g2 = ThreeAnd.f(eq1, value3, b1_n)
        o3 = TwoOr.f(g1, g2)
        return o3


class demux5_o1:
    name = "demux5_o1"
    numInputs = 4
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value1, value2, value3, value4):
        z1 = OneNot.f(value2)
        z2 = OneNot.f(value3)
        z3 = OneNot.f(value4)
        return FourAnd.f(value1, z1, z2, z3)

class demux5_o2:
    name = "demux5_o2"
    numInputs = 4
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value1, value2, value3, value4):
        z1 = OneNot.f(value2)
        z2 = OneNot.f(value3)
        return FourAnd.f(value1, z1, z2, value4)

class demux5_o3:
    name = "demux5_o3"
    numInputs = 4
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value1, value2, value3, value4):
        z1 = OneNot.f(value2)
        z3 = OneNot.f(value4)
        return FourAnd.f(value1, z1, value3, z3)

class demux5_o4:
    name = "demux5_o4"
    numInputs = 4
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value1, value2, value3, value4):
        z1 = OneNot.f(value2)
        return FourAnd.f(value1, z1, value3, value4)

class demux5_o5:
    name = "demux5_o5"
    numInputs = 4
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value1, value2, value3, value4):
        z2 = OneNot.f(value3)
        z3 = OneNot.f(value4)
        return FourAnd.f(value1, value2, z2, z3)


class demux6_o6:
    name = "demux6_o6"
    numInputs = 4
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value1, value2, value3, value4):
        z2 = OneNot.f(value3)
        return FourAnd.f(value1, value2, z2, value4)


class mul2_o1:
    name = "mul2_o1"
    numInputs = 4
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value1, value2, value3, value4):
        return TwoAnd.f(value1, value3)

class mul2_o2:
    name = "mul2_o2"
    numInputs = 4
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value1, value2, value3, value4):
        a_1_1 = TwoAnd.f(value2, value3)
        a_2_2 = TwoAnd.f(value1, value4)
        return TwoAnd.f(a_1_1, a_2_2)

class mul2_o3:
    name = "mul2_o3"
    numInputs = 4
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value1, value2, value3, value4):
        a_1_1 = TwoAnd.f(value2, value3)
        a_2_1 = TwoAnd.f(value2, value4)
        a_2_2 = TwoAnd.f(value1, value4)
        c_2_2 = TwoAnd.f(a_1_1, a_2_2)
        return TwoXor.f(c_2_2, a_2_1)

class mul2_o4:
    name = "mul2_o4"
    numInputs = 4
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value1, value2, value3, value4):
        a_1_1 = TwoAnd.f(value2, value3)
        a_2_1 = TwoAnd.f(value2, value4)
        a_2_2 = TwoAnd.f(value1, value4)
        c_2_2 = TwoAnd.f(a_1_1, a_2_2)
        return TwoAnd.f(c_2_2, a_2_1)


class multi_operand_adder4_o1:
    name = "multi_operand_adder4_o1"
    numInputs = 4
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value1, value2, value3, value4):
        l1_z1 = TwoXor.f(value2, value1)
        l2_z1 = TwoXor.f(value3, l1_z1)
        return TwoXor.f(value4, l2_z1)


class multi_operand_adder4_o2:
    name = "multi_operand_adder4_o2"
    numInputs = 4
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value1, value2, value3, value4):
        l1_z1 = TwoXor.f(value2, value1)
        l1_z2 = TwoAnd.f(value2, value1)
        l2_z1 = TwoXor.f(value3, l1_z1)
        l2_c1 = TwoAnd.f(value3, l1_z1)
        l2_z2 = TwoXor.f(l1_z2, l2_c1)
        l3_c1 = TwoAnd.f(value4, l2_z1)
        return TwoXor.f(l3_c1, l2_z2)


class multi_operand_adder4_o3:
    name = "multi_operand_adder4_o3"
    numInputs = 4
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value1, value2, value3, value4):
        l1_z1 = TwoXor.f(value2, value1)
        l1_z2 = TwoAnd.f(value2, value1)
        l2_z1 = TwoXor.f(value3, l1_z1)
        l2_c1 = TwoAnd.f(value3, l1_z1)
        l2_z2 = TwoXor.f(l1_z2, l2_c1)
        l3_c1 = TwoAnd.f(value4, l2_z1)
        return TwoAnd.f(l3_c1, l2_z2)


class subtractor2_o1:
    name = "subtractor2_o1"
    numInputs = 5
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value1, value2, value3, value4, value5):
        S1_f = TwoXor.f(value1, value2)
        return TwoXor.f(S1_f, value5)


class subtractor2_o2:
    name = "subtractor2_o2"
    numInputs = 5
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value1, value2, value3, value4, value5):
        S1_HS1_z = OneNot.f(value1)
        S1_f = TwoXor.f(value1, value2)
        S1_p = TwoAnd.f(S1_HS1_z, value2)
        S1_HS2_z = OneNot.f(S1_f)
        S1_q = TwoAnd.f(S1_HS2_z, value5)
        bi1 = TwoOr.f(S1_p, S1_q)
        S2_f = TwoXor.f(value3, value4)
        return TwoXor.f(S2_f, bi1)


class subtractor2_o3:
    name = "subtractor2_o3"
    numInputs = 5
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value1, value2, value3, value4, value5):
        S1_HS1_z = OneNot.f(value1)
        S1_f = TwoXor.f(value1, value2)
        S1_p = TwoAnd.f(S1_HS1_z, value2)
        S1_HS2_z = OneNot.f(S1_f)
        S1_q = TwoAnd.f(S1_HS2_z, value5)
        bi1 = TwoOr.f(S1_p, S1_q)
        S2_HS1_z = OneNot.f(value3)
        S2_f = TwoXor.f(value3, value4)
        S2_p = TwoAnd.f(S2_HS1_z, value4)
        S2_HS2_z = OneNot.f(S2_f)
        S2_q = TwoAnd.f(S2_HS2_z, bi1)
        return TwoOr.f(S2_p, S2_q)


class mux4_o1:
    name = "mux4_o1"
    numInputs = 6
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value1, value2, value3, value4, value5, value6):
       z1 = OneNot.f(value1)
       z2 = OneNot.f(value2)
       q1 = ThreeAnd.f(value3, z1, z2)
       q2 = ThreeAnd.f(value4, z1, value2)
       q3 = ThreeAnd.f(value5, value1, z2)
       q4 = ThreeAnd.f(value6, value1, value2)
       return FourOr.f(q1, q2, q3, q4)


class comp3_o1:
    name = "comp3_o1"
    numInputs = 6
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value1, value2, value3, value4, value5, value6):
        eq1 = OneNot.f(TwoXor.f(value1, value2))
        eq2 = OneNot.f(TwoXor.f(value3, value4))
        eq3 = OneNot.f(TwoXor.f(value5, value6))
        eq = ThreeAnd.f(eq1, eq2, eq3)
        b3_n = OneNot.f(value2)
        b2_n = OneNot.f(value4)
        b1_n = OneNot.f(value6)
        g1 = TwoAnd.f(value1, b3_n)
        g2 = ThreeAnd.f(eq1, value3, b2_n)
        g3 = FourAnd.f(eq1, eq2, value5, b1_n)
        gt = ThreeOr.f(g1, g2, g3)
        eq_gt = TwoOr.f(gt, eq)
        return OneNot.f(eq_gt)


class comp3_o2:
    name = "comp3_o2"
    numInputs = 6
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value1, value2, value3, value4, value5, value6):
        eq1 = OneNot.f(TwoXor.f(value1, value2))
        eq2 = OneNot.f(TwoXor.f(value3, value4))
        eq3 = OneNot.f(TwoXor.f(value5, value6))
        return ThreeAnd.f(eq1, eq2, eq3)


class comp3_o3:
    name = "comp3_o3"
    numInputs = 6
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value1, value2, value3, value4, value5, value6):
        eq1 = OneNot.f(TwoXor.f(value1, value2))
        eq2 = OneNot.f(TwoXor.f(value3, value4))
        b3_n = OneNot.f(value2)
        b2_n = OneNot.f(value4)
        b1_n = OneNot.f(value6)
        g1 = TwoAnd.f(value1, b3_n)
        g2 = ThreeAnd.f(eq1, value3, b2_n)
        g3 = FourAnd.f(eq1, eq2, value5, b1_n)
        return ThreeOr.f(g1, g2, g3)


class multi_operand_adder5_o1:
    name = "multi_operand_adder5_o1"
    numInputs = 5
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value1, value2, value3, value4, value5):
        l1_z1 = TwoXor.f(value2, value1)
        l2_z1 = TwoXor.f(value3, l1_z1)
        l3_z1 = TwoXor.f(value4, l2_z1)
        return TwoXor.f(value5, l3_z1)


class multi_operand_adder5_o2:
    name = "multi_operand_adder5_o2"
    numInputs = 5
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value1, value2, value3, value4, value5):
        l1_z1 = TwoXor.f(value2, value1)
        l1_z2 = TwoAnd.f(value2, value1)
        l2_z1 = TwoXor.f(value3, l1_z1)
        l2_c1 = TwoAnd.f(value3, l1_z1)
        l2_z2 = TwoXor.f(l1_z2, l2_c1)
        l3_z1 = TwoXor.f(value4, l2_z1)
        l3_c1 = TwoAnd.f(value4, l2_z1)
        l3_z2 = TwoXor.f(l3_c1, l2_z2)
        l4_c1 = TwoAnd.f(value5, l3_z1)
        return TwoXor.f(l4_c1, l3_z2)


class multi_operand_adder5_o3:
    name = "multi_operand_adder5_o3"
    numInputs = 5
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value1, value2, value3, value4, value5):
        l1_z1 = TwoXor.f(value2, value1)
        l1_z2 = TwoAnd.f(value2, value1)
        l2_z1 = TwoXor.f(value3, l1_z1)
        l2_c1 = TwoAnd.f(value3, l1_z1)
        l2_z2 = TwoXor.f(l1_z2, l2_c1)
        l3_z1 = TwoXor.f(value4, l2_z1)
        l3_c1 = TwoAnd.f(value4, l2_z1)
        l3_z2 = TwoXor.f(l3_c1, l2_z2)
        l3_z3 = TwoAnd.f(l3_c1, l2_z2)
        l4_c1 = TwoAnd.f(value5, l3_z1)
        l4_c2 = TwoAnd.f(l4_c1, l3_z2)
        return TwoXor.f(l3_z3, l4_c2)