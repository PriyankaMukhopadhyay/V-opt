import numpy as np
import math
import time
import random

n=2
N=2**n
N2=4**n

G_Vp=[]
G_Vn=[]

# Define Pauli Matrices
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

# Create a list of Pauli matrices
P_1 = [I, X, Y, Z]

Vz_2 = np.array([[(1+2j)/math.sqrt(5),0,0,0],[0,(1-2j)/math.sqrt(5),0,0],[0,0,1,0],[0,0,0,1]])
Vz_3 = np.array([[1,0,0,0],[0,(1+2j)/math.sqrt(5),0,0],[0,0,(1-2j)/math.sqrt(5),0],[0,0,0,1]])
Vz_4 = np.array([[1,0,0,0],[0,1,0,0],[0,0,(1+2j)/math.sqrt(5),0],[0,0,0,(1-2j)/math.sqrt(5)]])

Vz_2_inv = Vz_2.conj().T
Vz_3_inv = Vz_3.conj().T
Vz_4_inv = Vz_4.conj().T

Vz = [Vz_2,Vz_2_inv,Vz_3,Vz_3_inv,Vz_4,Vz_4_inv]
print("Vz = ",Vz)

V2_2 = np.array([[1/math.sqrt(5),2/math.sqrt(5),0,0],[-2/math.sqrt(5),1/math.sqrt(5),0,0],[0,0,1,0],[0,0,0,1]])
V2_3 = np.array([[1,0,0,0],[0,1/math.sqrt(5),2/math.sqrt(5),0],[0,-2/math.sqrt(5),1/math.sqrt(5),0],[0,0,0,1]])
V2_4 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1/math.sqrt(5),2/math.sqrt(5)],[0,0,-2/math.sqrt(5),1/math.sqrt(5)]])

V2_2_inv = V2_2.conj().T
V2_3_inv = V2_3.conj().T
V2_4_inv = V2_4.conj().T

V2 = [V2_2, V2_2_inv, V2_3, V2_3_inv, V2_4, V2_4_inv]
print("V2 = ",V2)

V3_2 = np.array([[1/math.sqrt(5),2j/math.sqrt(5),0,0],[2j/math.sqrt(5),1/math.sqrt(5),0,0],[0,0,1,0],[0,0,0,1]])
V3_3 = np.array([[1,0,0,0],[0,1/math.sqrt(5),2j/math.sqrt(5),0],[0,2j/math.sqrt(5),1/math.sqrt(5),0],[0,0,0,1]])
V3_4 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1/math.sqrt(5),2j/math.sqrt(5)],[0,0,2j/math.sqrt(5),1/math.sqrt(5)]])

V3_2_inv = V3_2.conj().T
V3_3_inv = V3_3.conj().T
V3_4_inv = V3_4.conj().T

V3 = [V3_2, V3_2_inv, V3_3, V3_3_inv, V3_4, V3_4_inv]
print("V3 = ",V3)

V = [Vz_2,Vz_2_inv,Vz_3,Vz_3_inv,Vz_4,Vz_4_inv, V2_2, V2_2_inv, V2_3, V2_3_inv, V2_4, V2_4_inv, V3_2, V3_2_inv, V3_3, V3_3_inv, V3_4, V3_4_inv]

def MULTIQUBIT_PAULI(qubit):
        if qubit == 1:
            return P_1
        else:
            P_n = []
            for p in P_1:
              for q in MULTIQUBIT_PAULI(qubit-1):
                temp=np.kron(p,q)
                P_n.append(temp)
        return P_n
#            return [np.kron(p, q) for p in paulis for q in tensor_product_recursive(paulis, depth - 1)]

def tensor_product(operators):
    result = operators[0]
    for op in operators[1:]:
        result = np.kron(result, op)
    return result

P_n=MULTIQUBIT_PAULI(n)

print("P_n=",P_n)
print("size of P_n",len(P_n))

#-------------CHANNEL REP of GENERATING SET ELEMENTS-------------------------

G_V = []

for i in range(18):
  U = V[i]
  #print("U = ",U)
  G_Vz = []
  num0 = 0
  num1 = 0
  num2 = 0
  for r in range(len(P_n)):
    Pr = P_n[r]
    arr = []
    #print("Pr = ",Pr)
    U_temp = np.matmul(Pr,np.matmul(U,np.matmul(Pr,U.conj().T)))
    diag_trace = np.trace(U_temp)/N
    diag_r = diag_trace.real
    diag = round(diag_r,1)
    if diag == 0.2 :
      #arr.append(2)
      arr.append(r)
      num2 = num2+1
    if diag == 0.4 :
      #arr.append(1)
      arr.append(r)
      num1 = num1+1
    if diag == 1.0 :
      num0 = num0+1
    for s in range(len(P_n)):
      if s != r:
        Ps = P_n[s]
        U_temp = np.matmul(Pr,np.matmul(U,np.matmul(Ps,U.conj().T)))
        oDiag_trace = np.trace(U_temp)/N
        oDiag_r = oDiag_trace.real
        oDiag = round(oDiag_r,1)
        if (oDiag == 0.9) or (oDiag == 0.4):
          arr.append([s,2])
        if (oDiag == -0.9) or (oDiag == -0.4):
          arr.append([s,-2])
        if oDiag == 0.8:
          arr.append([s,4])
        if oDiag == -0.8:
          arr.append([s,-4])
        #if oDiag < 0.0 :
          #arr.append(-s)
        #if oDiag > 0:
          #arr.append(s)
    #print("arr = ",arr)
    #print("len = ",len(arr))
    if len(arr) != 0:
      G_Vz.append(arr)
  print("i,G_Vz = ",i,G_Vz)
  print("length of G_Vz = ",len(G_Vz))
  print("num0, num1, num2 = ",num0,num1,num2)
  G_V.append(G_Vz)

print("G_V = ",G_V)
print("length of G_V = ",len(G_V))

#-------------------ADD, SDE REDUCE, MULT--------------------------

def sde5_reduce(a,b,k): #a=rational, b=irrational, k=exponent, v=(a+b*sqrt(5))/ (sqrt(5))^k
  if (a == 0) and (b == 0):
    print("Bug ! a = b = 0")
  if (a == 0) and (b != 0):
    k = k-1
    a = b
    b = 0
  if (a != 0) and (b == 0): #Only rational. sde reduces by 2 with every reduction
    while a % 5 == 0 and a != 0 and k!=0:  # Check if 'a' is divisible by 5 and not zero
      a = a // 5  # Use integer division to avoid floating point result
      k = k - 2
      if k < 0 :
        print("Error : a,b,k = ",a,b,k)
      if (k==0) and (a != 1):
        print("Error : a,b,k = ",a,b,k)
  if (a != 0) and (b != 0):
    while a % 5 == 0:
      temp = a // 5
      a = b
      b = temp
      k = k-1

  return [a,b,k]

def add_5(v1, v2):
  [a1,b1,k1], [a2,b2,k2] = v1, v2

  if k1 == k2:
    a = a1+a2
    b = b1+b2
    if (a == 0) and (b == 0):
      k = 0
    else:
      k = k1

  if k1 > k2:
    diff = k1-k2
    if diff % 2 == 0:
      m = diff // 2
      a = a1 + a2*(5**m)
      b = b1 + b2*(5**m)
      if (a == 0) and (b == 0):
        k = 0
      else:
        k = k1
    else:
      m = (diff-1) // 2
      a = a1 + b2*(5**(m+1))
      b = b1 + a2*(5**m)
      if (a == 0) and (b == 0):
        k = 0
      else:
        k = k1

  if k1 < k2:
    diff = k2-k1
    if diff % 2 == 0:
      m = diff // 2
      a = a2 + a1*(5**m)
      b = b2 + b1*(5**m)
      if (a == 0) and (b == 0):
        k = 0
      else:
        k = k2
    else:
      m = (diff-1) // 2
      a = a2 + b1*(5**(m+1))
      b = b2 + a1*(5**m)
      if (a == 0) and (b == 0):
        k = 0
      else:
        k = k2

  if (a == 0) and (b == 0):
    return [0,0,0]
  else:
    return sde5_reduce(a,b,k)

# Example usage:
a = 4*(5**4)
b = 3*(5**2)
k = 20
print("sde_5 reduce=",sde5_reduce(a,b,k))
# Example usage:
# Assuming we have fractions (2, 1) equivalent to 2/5 and (3, 2) equivalent to 3/25
v1 = (0,0,0)
v2 = (2,0,2)

result = add_5(v1, v2)
print(result)  # Output would depend on the implementation of sde5_REDUCE

def MULT_Gv(P, U):
    # Input P is a number from 0 to 8.
  Up = [[[0,0,0] for _ in range(N2)] for _ in range(N2)]
  for i in range(N2):
    for j in range(N2):
      Up[i][j] = U[i][j]

  G_Vz = G_V[P]
  for i in range(12):
    arr = G_Vz[i]
    size_arr = len(arr)
    diag = arr[0]
    if size_arr == 2:
      oDiag = arr[1][0]
      const = arr[1][1]
      for j in range(N2):
        v1 = U[diag][j]
        v2 = U[oDiag][j]
        if (v1[0] == 0) and (v1[1] == 0):
          v1 = [0,0,0]
        else:
          v1 = [v1[0], v1[1], v1[2]+1]
        if (v2[0] == 0) and (v2[1] == 0):
          v2 = [0,0,0]
        else:
          v2 = [const*v2[0], const*v2[1], v2[2]+1]
        Up[diag][j] = add_5(v1, v2)

    if size_arr == 4:
      oDiag1 = arr[1][0]
      const1 = arr[1][1]
      oDiag2 = arr[2][0]
      const2 = arr[2][1]
      oDiag3 = arr[3][0]
      const3 = arr[3][1]
      for j in range(N2):
        v1 = U[diag][j]
        v2 = U[oDiag1][j]
        v3 = U[oDiag2][j]
        v4 = U[oDiag3][j]
        if (v1[0] == 0) and (v1[1] == 0):
          v1 = [0,0,0]
        else:
          v1 = [v1[0], v1[1], v1[2]+2]
        if (v2[0] == 0) and (v2[1] == 0):
          v2 = [0,0,0]
        else:
          v2 = [const1*v2[0], const1*v2[1], v2[2]+2]
        if (v3[0] == 0) and (v3[1] == 0):
          v3 = [0,0,0]
        else:
          v3 = [const2*v3[0], const2*v3[1], v3[2]+2]
        if (v4[0] == 0) and (v4[1] == 0):
          v4 = [0,0,0]
        else:
          v4 = [const3*v4[0], const3*v4[1], v4[2]+2]
        #print("diag,j,v1,v2,v3,v4 = ",diag,j,v1,v2,v3,v4)
        sum12 = add_5(v1, v2)
        sum34 = add_5(v3, v4)
        Up[diag][j] = add_5(sum12, sum34)
        #print("sum12, sum34, sum = ",sum12, sum34, Up[diag][j])

  return Up

#------------------GET SDE, HAM-WT, UPDATE SH, MIN SH ---------------------------------------

def GET_SDE(U):

  sde=0

  for i in range(N2):
    for j in range(N2):
      if U[i][j][2] > sde:
          sde = U[i][j][2]

  return sde


def HAM_WT_MAT(U):
    ham = 0
    for i in range(N2):
        for j in range(N2):
            if (U[i][j][0] != 0) and (U[i][j][1] != 0):
                ham += 1
    return ham

# Example usage:
# U should be defined as an N^2 x N^2 matrix with appropriate values.
# For example:
#U = [[0, 1], [1, 0]] # For N=1
#hamming_weight = HAM_WT_MAT(U)
#print(hamming_weight)  # Output will be 2 if U is as defined above.


def UPDATE_SH(SH, sde1, ham1, sde0, ham0, rule):
    #s = h = 0  # Initialize s and h to 0 sde1 : child, sde0 : par
    # inc (0), unchanged (1), dec (2) : sde along row and ham along column
    #SH is always 3x3

    if rule == 1: # 9 groups : Both sde and ham inc, unchanged, dec
      if sde1 > sde0 and ham1 > ham0:
        SH[0][0] = SH[0][0] + 1; s = h = 0
      elif sde1 > sde0 and ham1 == ham0:
        SH[0][1] = SH[0][1] + 1; s = 0; h = 1
      elif sde1 > sde0 and ham1 < ham0:
        SH[0][2] = SH[0][2] + 1; s = 0; h = 2
      elif sde1 == sde0 and ham1 > ham0:
        SH[1][0] = SH[1][0] + 1; s = 1; h = 0
      elif sde1 == sde0 and ham1 == ham0:
        SH[1][1] = SH[1][1] + 1; s = 1; h = 1
      elif sde1 == sde0 and ham1 < ham0:
        SH[1][2] = SH[1][2] + 1; s = 1; h = 2
      elif sde1 < sde0 and ham1 > ham0:
        SH[2][0] = SH[2][0] + 1; s = 2; h = 0
      elif sde1 < sde0 and ham1 == ham0:
        SH[2][1] = SH[2][1] + 1; s = 2; h = 1
      elif sde1 < sde0 and ham1 < ham0:
        SH[2][2] = SH[2][2] + 1; s = 2; h = 2


    if rule == 2: # 4 groups : sde and ham inc or non-inc (dec+same)
      if sde1 > sde0 and ham1 > ham0:
        SH[0][0] = SH[0][0] + 1; s = h = 0
      elif sde1 > sde0 and ham1 <= ham0:
        SH[0][2] = SH[0][2] + 1; s = 0; h = 2
      elif sde1 <= sde0 and ham1 > ham0:
        SH[2][0] = SH[2][0] + 1; s = 2; h = 0
      elif sde1 <= sde0 and ham1 <= ham0:
        SH[2][2] = SH[2][2] + 1; s = 2; h = 2

    if rule == 3: #3 groups : sde inc, unchanged or dec
      if sde1 > sde0:
        SH[0][0] = SH[0][0] + 1; s = h = 0
      elif sde1 == sde0:
        SH[1][0] = SH[1][0] + 1; s = 1; h = 0
      elif sde1 < sde0:
        SH[2][0] = SH[2][0] + 1; s = 2; h = 0

    if rule == 4: #2 groups : sde inc, non-inc (same+dec)
      if sde1 > sde0:
        SH[0][0] = SH[0][0] + 1; s = h = 0
      elif sde1 <= sde0:
        SH[2][0] = SH[2][0] + 1; s = 2; h = 0

    return (SH, s, h)

# Example usage:
# SH should be defined as a state with appropriate values.
# For example:
# SH = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
# sde0, sde1 = 0, 1
# ham0, ham1 = 1, 2
# new_SH, s, h = UPDATE_SH(SH, sde1, ham1, ham0)
# print(new_SH, s, h)

def MIN_SH(SH):

  start = 0
  #print("Input SH = ",SH)
  for i in range(3):
    for j in range(3):
      if (start == 0) and (SH[i][j] > 0) :
        start = 1
        min_val = SH[i][j]
        s_index = i
        h_index = j
      if (start == 1) and (SH[i][j] > 0):
        if SH[i][j] <= min_val:
          min_val = SH[i][j]
          s_index = i
          h_index = j

  return (s_index, h_index)

#----------------EXACT-V-DECIDE-------------------------------------

def EXACT_V_DECIDE(U, m, sde_U):

  ham_U=HAM_WT_MAT(U)
  Path_U=[]
  Par_node=[]
  U_tilde=[U,Path_U,sde_U,ham_U]
  #print("U_tilde",U_tilde)
  Par_node.append(U_tilde)
  #print("Root = ",Par_node)
  SH=np.zeros((3,3),dtype=int)
  rule = 1
  leaf = 0
  max_sel_node = 1

  for i in range(1,m+1):
    if leaf == 1:
      #print("Breaking due to leaf==1 at i = ",i)
      break
    Child_node=[]
    print("Level = ",i)
    num_par = len(Par_node)
    if num_par > max_sel_node:
      max_sel_node = num_par
    #print(" No of Parents = ",len(Par_node))
    SH=np.zeros((3,3),dtype=int)
    if num_par == 0:
      #print("No of parent nodes is ..so breaking",len(Par_node))
      break
    for j in range(num_par):
      if leaf == 1:
        #print("Breaking due to leaf ==1 at j = ",j)
        break
      U_par=Par_node[j][0]
      #print("U+par = ",U_par)
      Path_par=Par_node[j][1]
      #print("Path_par = ",Path_par)
      sde_par=Par_node[j][2]
      ham_par=Par_node[j][3]
      path_len=len(Path_par)
      if path_len > 0:
        P_prev=Path_par[path_len-1]
      for k in range(18):
        Path_W=[]
        P=k
        if path_len > 0:
          for p_c in range(path_len):
            Path_W.append(Path_par[p_c])
        if (path_len > 0) and (P == -P_prev):
          continue
        else:
          #print("P=",P)
          W=MULT_Gv(P,U_par)
          sde_W=GET_SDE(W)
          #print("P,sde_W,m+1-i = ",P,sde_W)
          #print("m+1-i=",m+1-i)
          ham_W=HAM_WT_MAT(W)
          Path_W.append(P)
          if sde_W == 0:
            fin_i = i
            fin_Path = Path_W
            leaf = 1
            break
          #if sde_W > m+1-i:
            #print("P,sde_W,m+1-i, case 2=",P,sde_W,m+1-i)
            #continue
          else:
            new_SH=UPDATE_SH(SH,sde_W,ham_W,sde_par,ham_par,rule)
            #print("P,sde_W,sde_par,m+1-i, Case 3 = ",P,sde_W,sde_par,m+1-i)
            SH=new_SH[0]
            s_W=new_SH[1]
            h_W=new_SH[2]
            Child_node.append([W,s_W,h_W,Path_W,sde_W,ham_W])


    num_child=len(Child_node)
    if (leaf == 0) and (num_child != 0):
      #print("Children = ",Child_node)
      #print("No of children = ",len(Child_node))
      sh=MIN_SH(SH)
      s_indx=sh[0]
      h_indx=sh[1]
      #print("SH,s_indx,h_index = ",SH,s_indx,h_indx)
      #print("h_indx = ",h_indx)
      #print("SH=",SH)
    Par_node = []

    if num_child != 0:
      for j in range(len(Child_node)):
        if leaf == 1:
          print("breaking due to leaf == 1 at child j,",j)
          break
        if Child_node[j][4] > m+1-i:
            #print("P,sde_W,m+1-i, case 2=",P,sde_W,m+1-i)
            continue
        if Child_node[j][4] == 1:
          next_U=Child_node[j][0]
          next_U_Path=Child_node[j][3]
          next_U_sde=Child_node[j][4]
          next_U_ham=Child_node[j][5]
          Par_node.append([next_U,next_U_Path,next_U_sde,next_U_ham])
        if (Child_node[j][1] == s_indx) and (Child_node[j][2] == h_indx):
          next_U=Child_node[j][0]
          next_U_Path=Child_node[j][3]
          next_U_sde=Child_node[j][4]
          next_U_ham=Child_node[j][5]
          Par_node.append([next_U,next_U_Path,next_U_sde,next_U_ham])

  if leaf == 1:
    print("Max no of selected nodes = ",max_sel_node)
    return (fin_i,fin_Path)
  else :
    return (-1,[])


#---------------SPECIFIC UNITARIES--------------------------------------

#IVz = np.array([[(1+2j)/np.sqrt(5),0,0,0],[0,(1-2j)/np.sqrt(5),0,0],[0,0,(1+2j)/np.sqrt(5),0],[0,0,0,(1-2j)/np.sqrt(5)] ]) #id\otimes V_z
#IVz = np.array([[1/np.sqrt(5),(2j)/np.sqrt(5),0,0],[(2j)/np.sqrt(5),1/np.sqrt(5),0,0],[0,0,1/np.sqrt(5),(2j)/np.sqrt(5)],[0,0,(2j)/np.sqrt(5),1/np.sqrt(5)] ]) #id\otimes V_x
IVz = np.array([[1/np.sqrt(5),(2)/np.sqrt(5),0,0],[(-2)/np.sqrt(5),1/np.sqrt(5),0,0],[0,0,1/np.sqrt(5),(2)/np.sqrt(5)],[0,0,(-2)/np.sqrt(5),1/np.sqrt(5)] ]) #id\otimes V_y

print("IVz = ",IVz)
IVz_adj = IVz.conj().T
print("IVz_adj = ",IVz_adj)

U_temp =[[[0,0,0] for i in list(range(N2))] for j in list(range(N2))]
diag = 0
noDiag = 0
poDiag = 0


for i in range(N2):
  Pr = P_n[i]
  #print("Pr = ",Pr)
  prod = np.matmul(Pr,np.matmul(IVz,np.matmul(Pr,IVz_adj)))
  val = np.trace(prod)/N
  val_real = val.real
  val_r = round(val_real,1)
  if val_r == 1.0:
    U_temp[i][i] = [1,0,0]
  elif val_r == -0.6:
    U_temp[i][i] = [-3,0,2]
    diag = diag+1
  else:
    print("Err in diagonal, val_r = ",val_r)
  for j in range(N2):
    if j == i:
      continue
    Ps = P_n[j]
    #print("Ps = ",Ps)
    prod = np.matmul(Pr,np.matmul(IVz,np.matmul(Ps,IVz_adj)))
    val = np.trace(prod)/N
    val_real = val.real
    val_r = round(val_real,1)
    if val_r == 0.8 :
      U_temp[i][j] = [4,0,2]
      poDiag = poDiag+1
    elif val_r == -0.8:
      U_temp[i][j] = [-4,0,2]
      noDiag = noDiag+1
    elif val_r == 0.0:
      continue
    else:
      print("Error in off-diag, val_r = ",val_r)



print("U_in = ",U_in)
print("diag, poDiag, noDiag, sum = ",diag, poDiag, noDiag, diag+poDiag+noDiag)

#--------------------RANDOM-CHAN-REP----------------------------------

# Code for Random Unitaries
v_in = 10

U_temp =[[[0,0,0] for i in list(range(N2))] for j in list(range(N2))]
for i in range(N2):
  U_temp[i][i]=[1,0,0]

print("U_temp = ",U_temp)
sde_in=0
i=0

while i < v_in:
  P_in = np.random.randint(0, 18)
  #P_in = 0
  print("P_in = ",P_in)
  U_temp = MULT_Gv(P_in, U_temp)
  i = i+1

sde_in=GET_SDE(U_temp)
print("Final sde = ",sde_in)
print("U_temp = ",U_temp)

#-----------------EXACT-V-OPT------------------------

#EXACT-V-OPT
  #Input channel rep
print("Input Unitary = ", U_temp)
sde_in = GET_SDE(U_temp)
print("Input sde = ", sde_in)
m = sde_in
reach = 0

if m == 0:
    print("Not Clifford")

start_time = time.time()
while reach == 0:
    m_prime, D = EXACT_V_DECIDE(U_temp,m,sde_in)
    print("m_prime, D = ",m_prime,D)
    if m_prime == -1:
      m += 1
      #reach = 1
    else:
      print("m_prime, D = ", m_prime,D)
      print("Path length = ",len(D))
      print("Input Unitary = ", U_temp)
      reach = 1

execTime = time.time()-start_time
print("Time = ",execTime)













