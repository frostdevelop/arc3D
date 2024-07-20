# https://stackoverflow.com/questions/432112/is-there-a-numpy-function-to-return-the-first-index-of-something-in-an-array
import numpy as np
from numba import njit

class hi:
  def __init__(self,a,b):
    self.a = a
    self.b = b

  def mult(g):
    return self.a*g

  @staticmethod
  @njit()
  def allmult(g,mult):
    return mult(g)

hic = hi(1,2)
print(hic.allmult(12,hic.mult))