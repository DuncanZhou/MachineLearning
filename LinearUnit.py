#!/usr/bin/python

from Perception import Perception
import matplotlib.pyplot as plt
#f is the activator fuctiond

f = lambda x : x

class LinearUnit(Perception):
	def __init__(self,input_num):
		"""init LinearUnit"""
		Perception.__init__(self,input_num,f)

def get_training_dataset():
	#the work year
	input_vecs = [[5],[3],[8],[1.4],[10.1]]
	#labels is the salary
	labels = [5500,2300,7600,1800,11400]
	return input_vecs,labels

def train_linear_unit():
	#create a perception,input-parameter is the work year
	lu = LinearUnit(1)
	input_vecs,labels = get_training_dataset()
	#iteration is 10,rate is 0.01
	lu.train(input_vecs,labels,10,0.01)
	return lu

if __name__ == '__main__':
	linear_unit = train_linear_unit()
	print linear_unit
	print 'work 3.4 year,monthly salary = %.2f' % linear_unit.predict([3.4])
	print 'work 15 year,monthly salary = %.2f' % linear_unit.predict([15])
	print 'work 1.5 year,monthly salary = %.2f' % linear_unit.predict([1.5])
	print 'work 6.3 year,monthly salary = %.2f' % linear_unit.predict([6.3])
	x =[]
	y = []
	for i in range(12):
		x.append(i)
		y.append(i * linear_unit.weights[0])
	plt.plot(x,y)
	plt.scatter([3.4,15,1.5,6.3],[linear_unit.predict([3.4]),linear_unit.predict([15]),linear_unit.predict([1.5]),linear_unit.predict([6.3])],marker = '.')
	plt.show()


