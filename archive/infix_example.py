# https://web.archive.org/web/20220528202902/https://code.activestate.com/recipes/384122/
def funk(a,b):
    return a * b

class Infix:
    def __init__(self, function):
        self.function = function
    def __ror__(self, other):
        return Infix(lambda x, self=self, other=other: self.function(other, x))
    def __or__(self, other):
        return self.function(other)
    
dot=Infix(funk)
x=Infix(funk)

print(2 |dot| 3)
print(2 |x| 3)