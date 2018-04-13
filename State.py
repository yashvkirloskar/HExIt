import numpy as np

class State:
	def __init__(self, n):
		self.n = n
	def __eq__(self, other):
		return self.n == other.n
	def __repr__(self):
		return str(self.n)