#  Copyright (c) 2019.
#  Author:  Tao-Yi Lee
#  This source code is released under MIT license. Check LICENSE for details.


class SelectedIndexes:
    _I = None
    _I_bar = None

    def __init__(self, n: int):
        self.n = n
        self._I = []
        assert self.IBar

    def add(self, i):
        self._I.append(i)
        self._I.sort()
        self._I_bar.remove(i)

    def remove(self, i):
        self._I_bar.append(i)
        self._I_bar.sort()
        self._I.remove(i)

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
