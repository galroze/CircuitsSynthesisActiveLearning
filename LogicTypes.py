import operator


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