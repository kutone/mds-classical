import numpy as np
import matplotlib.pyplot as plt
import math

def mds():
  D = np.loadtxt('distances.txt', delimiter=',')
  C = np.genfromtxt('cities.txt', dtype = 'str', delimiter="\n")
  n = len(D)
  H = np.eye(n) - np.ones((n,n))/n
  B = -0.5 * H.dot(D**2).dot(H)
 
  evals, evecs = np.linalg.eigh(B)
  idx = np.argsort(evals)[::-1]
  evals = evals[idx]
  evecs = evecs[:,idx] # col i represents eigenvector i

  #w, = np.where(evals > 0)
  w = [0,1]
  L = np.diag(np.sqrt(evals[w]))
  U = evecs[:,w]
  Y = U.dot(L)
  return Y, C

def plot(Y, C, d):
  fig = plt.figure(0)
  ax = fig.add_subplot(111)
  ax.scatter(Y[:,0], Y[:,1])
  for i in range(len(C)):
    ax.annotate(C[i], (Y[i][0], Y[i][1]+20), fontsize = 10)
  plt.savefig('hw7-4.pdf')
  fig1 = plt.figure(1)
  theta = d / 180.0 * math.pi
  R = np.array([[math.cos(theta), -math.sin(theta)],[math.sin(theta), math.cos(theta)]])
  Y_t = R.dot(Y.T).T
  ax1 = fig1.add_subplot(111)
  ax1.scatter(Y_t[:,0], Y_t[:,1])
  for i in range(len(C)):
    ax1.annotate(C[i], (Y_t[i][0], Y_t[i][1]+20), fontsize = 10)
  plt.savefig('hw7-4_turn.pdf')

def main():
  Y, C = mds()
  plot(Y, C, 175)

if __name__ == '__main__':
  main()
