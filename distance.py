import numpy as np

def cosine(XA, XB):
  XA = np.array(XA)
  XB = np.array(XB)
  return 1 - np.dot(XA, XB) / (np.sqrt(np.dot(XA, XA)) *  np.sqrt(np.dot(XB, XB)))

def euclidean(XA, XB):
  XA = np.array(XA)
  XB = np.array(XB)
  return np.sqrt(np.sum(np.square(XA - XB)))

def manhattan(XA, XB):
  XA = np.array(XA)
  XB = np.array(XB)
  return np.sum(np.abs(XA - XB))

