import numpy as np
import math
from math import pi
import time
import random

n=2
N=2**n
N2=4**n
epsilon = 0.05
r_dig = 6 #Rounding place

# Define the matrices
I = np.array([[1, 0], [0, 1]])
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
T = np.array([[1,0],[0, np.exp((1j*pi)/4)]])

def tensor_product_recursive(paulis, depth):
        if depth == 1:
            return paulis
        else:
            return [np.kron(p, q) for p in paulis for q in tensor_product_recursive(paulis, depth - 1)]

def tensor_product(operators):
    result = operators[0]
    for op in operators[1:]:
        result = np.kron(result, op)
    return result

def trace_product(W_prime, P_out, P_in):
    product = np.matmul(W_prime, np.matmul(P_out, np.matmul(W_prime.conj().T, P_in)))
    return np.trace(product)

def matrix_product_recursive(S, depth):
        if depth == 1:
            return S
        else:
            return [np.matmul(p, q) for p in S for q in matrix_product_recursive(S, depth - 1)]

def matrix_neg_check(A, B):
    for i in range(N):
        for j in range(N):
            if A[i][j] != -B[i][j]:
                return False
    return True

def matrix_eq_check(A,B): #Returns 1 if equal
    eq = 1
    for i in range(N):
      if eq == 0:
        break
      for j in range(N):
        #if (A[i][j] - B[i][j] < 10**-6) or (B[i][j] - A[i][j] < 10**-6):
        if (A[i][j] - B[i][j] == 0):
          eq = 1
        else:
          eq = 0
          break
    return eq

def generate_pauli_n(n):
    """Generate the set of Pauli matrices for n qubits."""
    # Base Pauli matrices
    I = np.array([[1, 0], [0, 1]])
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    T = np.array([[1,0],[0, np.exp((1j*pi)/4)]])
    paulis = [I, X, Y, Z]

    return tensor_product_recursive(paulis, n)

pauli_I = [I]
i_tensored = tensor_product_recursive(pauli_I, n)[0]
pauli_n = generate_pauli_n(n)

total_pauli = []  #Positive and negative no-identity Paulis
for p in pauli_n:
  if np.allclose(p, i_tensored) == True:
    continue
  total_pauli.append(p)
  total_pauli.append(-p)

def CLIFF_TEST(W_prime): #Returns 1 if Clifford

  i = 0
  cliff = 1
  for P in pauli_n:
    result = np.matmul(W_prime, P)
    tr_result = np.abs(np.trace(result) / N)
    if (tr_result != 0) and (i == 0):
      if tr_result == 1:
        break
      else:
        prev_amp = tr_result
        i = i+1
    if (tr_result != 0) and (i != 0):
      if tr_result != prev_amp:
        print("Not Clifford after amplitude test.")
        cliff = 0
        return 0

  for P_out in pauli_n:
    if cliff == 0:
      print("Exiting outer loop of conjugation test")
      return 0
    if np.allclose(P_out, i_tensored) == True:
      #print("Got I!")
      continue

    for P_in in pauli_n:
      if np.allclose(P_in, i_tensored) == True:
        #print("Got I!")
        continue
      val = abs(trace_product(W_prime, P_out, P_in)) / N
      if val == 1:
        break
      if (val != 1) and (val != 0):
        print("Not Clifford after conjugation test.")
        cliff = 0
        break

  if cliff == 1:
    print("Passed Clifford test")
    return 1

b_conj = 1 - 4 * epsilon**2 + 2 * epsilon**4
print("b_conj = ",b_conj)
#b_conj = round(b_conj,r_dig)
print("b_conj after rounding  = ",b_conj)

l_conj = 2 * epsilon
print("l_conj = ",l_conj)
#l_conj = round(l_conj,r_dig)
print("l_conj after rounding = ",l_conj)

def A_CONJ(W_prime, epsilon):

    p = 1

    for P_out in pauli_n:
        if np.allclose(P_out, i_tensored) == True:
            #print("Got I!")
            continue
        if p == 1:
            p = 0

        for P_in in pauli_n:
            if np.allclose(P_in, i_tensored) == True:
                #print("Got I!")
                continue
            val = abs(trace_product(W_prime, P_out, P_in)) / N
            val = round(val,r_dig)
            #b_conj = 1 - 4 * epsilon**2 + 2 * epsilon**4

            if b_conj <= val <= 1:
                p += 1
                if p > 1:
                    return "NO"

            if l_conj < val < b_conj:
                return "NO"

    return "YES"

LB1 = []
UB1 = []
UB0 = []

for M in range(1,N2+1):
  lb_1 = (1 - epsilon**2) / np.sqrt(M) - np.sqrt(M * (2 * epsilon**2 - epsilon**4))
  LB1.append(lb_1)
  ub_1 = (1 / np.sqrt(M)) + np.sqrt(M * (2 * epsilon**2 - epsilon**4))
  UB1.append(ub_1)
  ub_0 = np.sqrt(M * (2 * epsilon**2 - epsilon**4))
  UB0.append(ub_0)

print("LB1 = ",LB1)
print("UB1 = ",UB1)
print("UB0 = ",UB0)

#for M in range(0,N2):
  #LB1[M] = round(LB1[M],r_dig)
  #UB1[M] = round(UB1[M],r_dig)
  #UB0[M] = round(UB0[M],r_dig)

print("LB1 after rounding = ",LB1)
print("UB1 after rounding = ",UB1)
print("UB0 after rounding = ",UB0)

def A_DECIDE(W,U_tilde, epsilon):

  W_prime = np.matmul(np.conj(W.T), U_tilde)
  #tr_Wprime = np.abs(np.trace(W_prime) / N)
  #dist = math.sqrt(1-tr_Wprime)
  S_c = []

  for P in pauli_n:
    result = np.matmul(W_prime, P)
    tr_result = np.abs(np.trace(result) / N)
    tr_result = round(tr_result,r_dig)
    S_c.append(tr_result)

  S_c.sort(reverse=True)

  for M in range(1,4**n+1):
      S_1 = S_c[:M]
      S_0 = S_c[M:]
      #lb1 = (1 - epsilon**2) / np.sqrt(M) - np.sqrt(M * (2 * epsilon**2 - epsilon**4))
      #ub1 = (1 / np.sqrt(M)) + np.sqrt(M * (2 * epsilon**2 - epsilon**4))
      #ub0 = np.sqrt(M * (2 * epsilon**2 - epsilon**4))
      amp = 1
      for x in S_1:
        if LB1[M-1] <= x <= UB1[M-1]:
          amp = 1
        else:
          amp = 0
          break
        if amp == 1:
          for x in S_0:
            if 0 <= x <= UB0[M-1]:
              amp = 1
            else:
              amp = 0
              break
        if amp == 1:
          print("Conj begin")
          result = A_CONJ(W_prime, epsilon)
          if result == "YES":
            execTime = time.time()-start_time
            #print("Fin Utilde",U_tilde)
            #print("Fin S_c",S_c)
            #print("Fin M",M)
            #print("Fin S_1",S_1)
            #print("Fin S_0",S_0)
            return "YES"

  return "NO"

#-----------------------------------SETS---------------------------------
S1 = []
S2 = []
S3 = []
S4 = []
S5 = []
S6 = []
S7 = []
S8 = []
S9 = []
S10 = []

for P1 in total_pauli:
  #print("P1", P1)
  U1 = 1/np.sqrt(5)*i_tensored + 1j*2/np.sqrt(5)*P1
  prod1 = U1
  S1.append(prod1)
  for P2 in total_pauli:
    if matrix_neg_check(P1, P2) == True:
      continue
    U2 = 1/np.sqrt(5)*i_tensored + 1j*2/np.sqrt(5)*P2
    prod2 = np.matmul(prod1, U2)
    S2.append(prod2)
    for P3 in total_pauli:
      if matrix_neg_check(P2, P3) == True:
        continue
      U3 = 1/np.sqrt(5)*i_tensored + 1j*2/np.sqrt(5)*P3
      prod3 = np.matmul(prod2, U3)
      S3.append(prod3)
      for P4 in total_pauli:
        if matrix_neg_check(P3, P4) == True:
          continue
        U4 = 1/np.sqrt(5)*i_tensored + 1j*2/np.sqrt(5)*P4
        prod4 = np.matmul(prod3, U4)
        S4.append(prod4)
        for P5 in total_pauli:
          if matrix_neg_check(P4, P5) == True:
            continue
          U5 = 1/np.sqrt(5)*i_tensored + 1j*2/np.sqrt(5)*P5
          prod5 = np.matmul(prod4, U5)
          S5.append(prod5)
          for P6 in total_pauli:
            if matrix_neg_check(P5, P6) == True:
              continue
            U6 = 1/np.sqrt(5)*i_tensored + 1j*2/np.sqrt(5)*P6
            prod6 = np.matmul(prod5, U6)
            S6.append(prod6)
            for P7 in total_pauli:
              if matrix_neg_check(P6, P7) == True:
                continue
              U7 = 1/np.sqrt(5)*i_tensored + 1j*2/np.sqrt(5)*P7
              prod7 = np.matmul(prod6, U7)
              S7.append(prod7)
              for P8 in total_pauli:
                if matrix_neg_check(P7, P8) == True:
                  continue
                U8 = 1/np.sqrt(5)*i_tensored + 1j*2/np.sqrt(5)*P8
                prod8 = np.matmul(prod7, U8)
                S8.append(prod8)
                for P9 in total_pauli:
                  if matrix_neg_check(P8, P9) == True:
                    continue
                  U9 = 1/np.sqrt(5)*i_tensored + 1j*2/np.sqrt(5)*P9
                  prod9 = np.matmul(prod8, U9)
                  S9.append(prod9)
                  for P10 in total_pauli:
                    if matrix_neg_check(P9, P10) == True:
                      continue
                    U10 = 1/np.sqrt(5)*i_tensored + 1j*2/np.sqrt(5)*P10
                    prod10 = np.matmul(prod9, U10)
                    S10.append(prod10)


size_S1 = len(S1)
size_S2 = len(S2)
size_S3 = len(S3)
size_S4 = len(S4)
size_S5 = len(S5)
size_S6 = len(S6)
size_S7 = len(S7)
size_S8 = len(S8)
size_S9 = len(S9)
size_S10 = len(S10)
print("Size of S1 = ",size_S1)
print("Size of S2 = ",size_S2)
print("Size of S3 = ",size_S3)
print("Size of S4 = ",size_S4)
print("Size of S5 = ",size_S5)
print("Size of S6 = ",size_S6)
print("Size of S7 = ",size_S7)
print("Size of S8 = ",size_S8)
print("Size of S9 = ",size_S9)
print("Size of S10 = ",size_S10)

#flag_match = 0
      #for i in range(len(S3)):
        #flag_match = matrix_eq_check(prod,S3[i],N)
        #if flag_match == 1:
        #  break
        #if flag_match == 0:
         # S3.append(prod)

def S_RECURSE(m):
  if m == 1:
    return S1
  else:
    nS = []
    for p in S1:
      prod1 = p[0]
      P_p = p[1]
      #print("P_p = ",P_p)
      U_p = 1/np.sqrt(5)*i_tensored + 1j*2/np.sqrt(5)*P_p
      for q in S_RECURSE(m-1):
        prod2 = q[0]
        P_q = q[1]
        #print("P_q = ",P_q)
        if matrix_neg_check(P_p, P_q) == True:
          #print("Mat check is True")
          continue
        prod_pq = np.matmul(prod2, U_p)
        nS.append([prod_pq,P_p])
  return nS
  
#----------------------OPT-------------------------


#W = np.array([[1,0],[0, np.exp((1j*pi)/4)]])  #T gate
theta_k = pow(2,6)  #k=0 : Z, k=1: S, k=2 : T
#W = np.array([[1,0],[0, np.exp((1j*pi)/theta_k)]])  #Rz gate
W = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0, np.exp((1j*pi)/theta_k)]])  #cRz gate
print("W = ",W)

cliff_result = CLIFF_TEST(W)
if cliff_result == 1:
  print("Found Clifford")
else:
  print("Not Clifford")

start_time = time.time()
succ = 0

i1 = 0

while i1  < size_S5:
  print("i1 = ",i1)
  i1 = i1+1
  if succ == 1:
    break
  U_1 = S5[i1]
  for i2 in range(size_S3):
    U_tilde = np.matmul(U_1, S3[i2])
    result = A_DECIDE(W,U_tilde, epsilon)
    if result == "YES":
      print("YES")
      succ = succ+1
      break



print("succ = ",succ)

execTime = time.time()-start_time
print("Implementation time = ",execTime)





