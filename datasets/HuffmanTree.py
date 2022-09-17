

class VocabWord:
    def __init__(self, word, count, index, nodes=None, code=None):

        self.index = index
        self.nodes = nodes
        self.code = code

        self.word = word
        self.cn = count
