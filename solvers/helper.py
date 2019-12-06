class SelectedIndexes:
    _I = None
    _I_bar = None

    def __init__(self, n: int):
        self.n = n
        self._I = []

    def add(self, i):
        self._I.append(i)
        self._I.sort()
        self._I_bar.remove(i)

    @property
    def I(self) -> list:
        return self._I

    @property
    def IBar(self) -> list:
        if self._I_bar is None:
            self._I_bar = list(set(list(range(self.n))) - set(self.I))
            self._I_bar.sort()
        return self._I_bar

    def __len__(self):
        return len(self._I)