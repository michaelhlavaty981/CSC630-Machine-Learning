import math
import numpy as np

def create_grad_arr(grad_position, total_vars):
    '''
    Creates gradient array with respect to a gradient component
    
    Parameters:
    grad_position = Int index of the gradient component in the array
    total_vars = Int representing the total number of Variable() created
    
    Returns:
    arr = numpy gradient array
    '''
    arr = np.zeros((total_vars,))
    arr[grad_position] = 1
    return arr

class Variable():
    vars = 0 #Class variable vars, used to keep track of the total number of Variable objects; index of gradient component = vars
    def __init__(self, name=None, eval_=None, grad=None, position = None):
        '''
        Initializes Variable objects
        
        Parameters:
        name = None, unless specified in Variable creation; used to identify and reference Variable objects
        eval_ = None, unless specified in the Variable creation; evaluate the Variable at a given value of primitive values
        position = None, unless specified in the Variable creation; used to locate the index of this Variable component when creating the gradient array
        '''
        if eval_ == None:
            self.eval_ = lambda values: values[self.name]
            self.position = Variable.vars
            Variable.vars += 1
        else:
            self.eval_ = eval_
            
        if name != None:
            self.name = name # its key in the evaluation dictionary
        
        if grad == None:
            self.grad = lambda values: create_grad_arr(self.position, Variable.vars)
        else:
            self.grad = grad

    
    def __call__(self, **kwargs):
        return self.eval_(kwargs)
            
    def __add__(self, other):
        '''
        Adds a Variable to either a float or int or Variable; defines functionality of the "+" operator

        Parameters:
        self - addend 1: Variable
        other - addend 2: Variable or int or float
        
        Returns:
        Variable that is the sum of self and other
        '''
        if isinstance(other, (int, float)):
            return Variable(eval_ = lambda values: self.eval_(values) + other,
                           grad = lambda values: self.grad(values),
                           position = Variable.vars)
            
        elif isinstance(other, Variable):
            return Variable(eval_ = lambda values: self.eval_(values) + other.eval_(values),
                           grad = lambda values: self.grad(values) + other.grad(values),
                           position = Variable.vars)
        else:
            return NotImplemented
    
    def __radd__(self, other):
        return self + other # calls self.__add__(other)
    
    def __sub__(self, other):
        return self + (-1 * other) # calls self.__add__(-1 * other)
    
    def __rsub__(self, other):
        return self * -1 + other # calls other.__add__(-1 * self) => (-1 * self).__add__(other)
    
    def __mul__(self, other):
        '''
        Multiplies a Variable to either a float or int or Variable; defines functionality of the "*" operator

        Parameters:
        self = Variable that is the multiplicand
        other = Variable or int or float that is the multiplier
        
        Returns:
        Variable that is the product of self and other
        '''
        if isinstance(other, (int, float)):
            return Variable(eval_ = lambda values: self.eval_(values) * other,
                           grad = lambda values: (self.grad(values) * other),
                           position = Variable.vars)
            
        elif isinstance(other,Variable):
            return Variable(eval_ = lambda values: self.eval_(values) * other.eval_(values),
                       grad = lambda values: (self.grad(values) * other.eval_(values) + other.grad(values) * self.eval_(values)),
                       position = Variable.vars)
        else:
            return NotImplemented
    
    def __rmul__(self, other):
        return self * other # calls self.__mul__(other)
    
    def __pow__(self, other):
        '''
        Raises a Variable to either a float or int or Variable; defines functionality of the "**" operator

        Parameters:
        self = Variable that is the base
        other = Variable or int or float that is the exponent
        
        Returns:
        Variable that is the product of self and other
        '''
        if isinstance(other, (int, float)):
            return Variable(eval_ = lambda values: self.eval_(values) ** other,
                           grad = lambda values: (other * (self.eval_(values) ** (other-1)) * self.grad(values)),
                           position = Variable.vars)
            
        elif isinstance(other, Variable):
            return Variable(eval_ = lambda values: self.eval_(values) ** other.eval_(values),
                       grad = lambda values: (other.eval_(values) * (self.eval_(values) ** (other.eval_(values)-1)) * self.grad(values)),
                       position = Variable.vars)
        else:
            return NotImplemented
    
    def __truediv__(self, other):
        return self * (other ** -1) # calls self.__mul__(other.__pow__(-1))
    
    def __rtruediv__(self, other):
        return (self ** -1) * other # calls self.__pow__(-1).__mul__(other)
    
def exp(self):
    '''
    Raises Euler's constant (e) to the power of a float or int or Variable

    Parameters:
    self = Variable or int or float that is the power

    Returns:
    Variable that is the product of e and self
    '''
    if isinstance(self, (int, float)):
        return Variable(eval_ = math.e ** self,
                       grad = (math.e ** self),
                       position = Variable.vars)

    return Variable(eval_ = lambda values: math.e ** self.eval_(values),
                   grad = lambda values: ((math.e ** self.eval_(values)) * self.grad(values)),
                   position = Variable.vars)

def log(self):
    '''
    Evaluates the natural logarithm of a Variable or int or float

    Parameters:
    self = Variable that is the argument

    Returns:
    Variable that is the logarithm of self with base e
    '''
    if isinstance(self, (int, float)):
        return Variable(eval_ = math.log(self),
                       grad = math.log(self))

    return Variable(eval_ = lambda values: math.log(self.eval_(values)),
                   grad = lambda values: (1 / self.eval_(values)) * self.grad(values),
                   position = Variable.vars)