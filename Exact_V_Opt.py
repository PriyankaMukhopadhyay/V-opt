import numpy as np
import math
import time
import random

n=1
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

def tensor_product(operators):
    result = operators[0]
    for op in operators[1:]:
        result = np.kron(result, op)
    return result

P_n=MULTIQUBIT_PAULI(n)

#print("P_n=",P_n)
print("size of P_n",len(P_n))

# Define a function to calculate the expression
def calculate_expression(P_r, P_s, P):
  val_mat= np.matmul(P_r,P_s)-2j*np.matmul(np.matmul(P_r, P_s), P) + 2j*np.matmul(np.matmul(P_r, P), P_s)+4*np.matmul(np.matmul(P_r, P), np.matmul(P_s, P))
  val_trace=np.trace(val_mat)
  val=val_trace/(5*N)
  return val

for i in range(0,N2):
  P=P_n[i]
  P1=-P
  G_VP=[]
  G_VP1=[]
  #print("P=",P)
  for j in range(N2):
    Pr=P_n[j]
    diag=-1
    diag1=-1
    val_c=calculate_expression(Pr,Pr,P)
    val0=val_c.real
    val=round(val0,1)
    val_c1=calculate_expression(Pr,Pr,P1)
    val01=val_c1.real
    val1=round(val01,1)
    if val==-0.6:
      diag=j
    if val1 == -0.6 :
      diag1=j
    for k in range(N2):
      if k != j:
        Ps=P_n[k]
        val_c=calculate_expression(Pr,Ps,P)
        val0=val_c.real
        val=round(val0,1)
        val_c1=calculate_expression(Pr,Ps,P1)
        val01=val_c1.real
        val1=round(val01,1)
        if val == 0.8:
          G_VP.append([diag,k])
        if val1 == 0.8 :
          G_VP1.append([diag1,k])
        if val == -0.8:
          G_VP.append([diag,-k])
        if val1 == -0.8:
          G_VP1.append([diag1,-k])
  G_Vp.append(G_VP)
  G_Vn.append(G_VP1)


print("length G_Vp", len(G_Vp))
print("length G_Vn", len(G_Vn))
print("G_Vp = ",G_Vp)
print("G_Vn = ",G_Vn)

#-----------------------OTHER SUB-ROUTINES----------------
def GET_SDE(U,Nsize):

  sde=0

  for i in range(Nsize):
    for j in range(Nsize):
      if U[i][j][1] > sde:
          sde = U[i][j][1]

  return sde

def sde5_reduce(a, k):
    while a % 5 == 0 and a != 0 and k!=0:  # Check if 'a' is divisible by 5 and not zero
        a = a // 5  # Use integer division to avoid floating point result
        #print(a)
        k = k - 1
        #print(k)
    return [a,k]

def add_5(v1, v2):
    (a1, k1), (a2, k2) = v1, v2

    if k1 >= k2:
        num = a1 + a2 * (5**(k1 - k2))
        if num == 0:
          den = 0
        else:
          den = k1
    else:
        num = a1 * (5**(k2 - k1)) + a2
        if num == 0:
          den = 0
        else:
          den = k2

    return sde5_reduce(num, den)

def MULT_Gv(P, U, N2):
    # Input P is a number from 0 to 4^n. s_in is sde of input matrix U.
    Up = [[[0, 0] for _ in range(N2)] for _ in range(N2)]

    for i in range(N2):
      for j in range(N2):
        Up[i][j] = U[i][j]

    #print("Input P = ",P)
    #print("Input Up=",Up)

    if P > 0:
      G_P = G_Vp[P]
      for i in range(int(N2/2)):
        diag=G_P[i][0]
        off_diag=G_P[i][1]
        if off_diag < 0:
          off_diag_abs=-off_diag
        else:
          off_diag_abs=off_diag
        for j in range(N2):
          v1 = U[diag][j]
          v2 = U[off_diag_abs][j]
          if v1[0] != 0:
            v1 = [-3 * v1[0], v1[1]+1]
          if off_diag < 0:
            if v2[0] != 0:
              v2 = [-4 * v2[0], v2[1]+1]
          else:
            if v2[0] != 0:
              v2 = [4 * v2[0], v2[1]+1]
          Up[diag][j] = add_5(v1, v2)
    else:
      G_P = G_Vn[-P]
      for i in range(int(N2/2)):
        diag=G_P[i][0]
        off_diag=G_P[i][1]
        if off_diag < 0:
          off_diag_abs=-off_diag
        else:
          off_diag_abs=off_diag
        for j in range(N2):
          v1 = U[diag][j]
          v2 = U[off_diag_abs][j]
          if v1[0] != 0:
            v1 = [-3 * v1[0], v1[1]+1]
          if off_diag < 0:
            if v2[0] != 0:
              v2 = [-4 * v2[0], v2[1]+1]
          else:
            if v2[0] != 0:
              v2 = [4 * v2[0], v2[1]+1]
          Up[diag][j] = add_5(v1, v2)

    #print("Product Up = ",Up)
    return Up

def HAM_WT_MAT(U,Nsize):
    ham = 0
    for i in range(Nsize):
        for j in range(Nsize):
            if U[i][j][0] != 0:
                ham += 1
    return ham

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

#-------------------EXACT-V-OPT----------------
def EXACT_V_DECIDE(U, m, Nsize, sde_U):

  ham_U=HAM_WT_MAT(U,Nsize)
  Path_U=[]
  Par_node=[]
  U_tilde=[U,Path_U,sde_U,ham_U]
  #print("U_tilde",U_tilde)
  Par_node.append(U_tilde)
  #print("Root = ",Par_node)
  SH=np.zeros((3,3),dtype=int)
  rule = 4
  leaf = 0
  max_sel_node = 1

  for i in range(1,m+1):
    if leaf == 1:
      #print("Breaking due to leaf==1 at i = ",i)
      break
    Child_node=[]
    num_par = len(Par_node)
    #print("Level, numpar = ",i,num_par)
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
      for k in range(1,N2):
        Path_W=[]
        P=k
        if path_len > 0:
          for p_c in range(path_len):
            Path_W.append(Path_par[p_c])
        if (path_len > 0) and (P == -P_prev):
          continue
        else:
          #print("P=",P)
          W=MULT_Gv(P,U_par,Nsize)
          sde_W=GET_SDE(W,Nsize)
          #print("P,sde_W,m+1-i = ",P,sde_W)
          #print("m+1-i=",m+1-i)
          ham_W=HAM_WT_MAT(W,Nsize)
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
      for k in range(1,N2):
        if leaf == 1:
          #print("Breaking at neg P at k = ",k)
          break
        P=-k
        Path_W=[]
        if path_len > 0:
          for p_c in range(path_len):
            Path_W.append(Path_par[p_c])
        if (path_len > 0) and (P == -P_prev):
          continue
        else:
          #print("P=",P)
          W=MULT_Gv(P,U_par,Nsize)
          sde_W=GET_SDE(W,Nsize)
          #print("P,sde_W = ",P,sde_W)
          #print("m+1-i=",m+1-i)
          ham_W=HAM_WT_MAT(W,Nsize)
          Path_W.append(P)
          if sde_W == 0:
            fin_i = i
            fin_Path = Path_W
            leaf = 1
            break
          #if sde_W > m+1-i:
            #print("P,sde_W,m+1-i, case 2=",P,sde_W,m+1-i)
           # continue
          else:
            new_SH=UPDATE_SH(SH,sde_W,ham_W,sde_par,ham_par,rule)
            #print("P,sde_W,sde_par,m+1-i, case 3 = ",P,sde_W,sde_par,m+1-i)
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

#-----------------Code for random unitaries-----------------
# Code for Random Unitaries
v_in = 10

U_temp =[[[0,0] for i in list(range(N2))] for j in list(range(N2))]
for i in range(N2):
  U_temp[i][i]=[1,0]

print("U_temp = ",U_temp)
sde_in=0
i=0

while i < v_in:
  P_in = np.random.randint(1, N2)
  b_in = np.random.randint(0, 2)
  if b_in == 1:
    P_in = -P_in

  print("P_in = ",P_in)

  if i == 0:
    P_prev = P_in
    U_temp = MULT_Gv(P_in, U_temp, N2)
    i = i+1
  else:
    if P_in == -P_prev:
      continue
    else:
      P_prev = P_in
      U_temp = MULT_Gv(P_in, U_temp, N2)
      i = i+1

sde_in=GET_SDE(U_temp,N2)
print("Final sde = ",sde_in)
print("U_temp = ",U_temp)

cols = list(range(0,N2))
print("cols = ",cols)
random.shuffle(cols)
print("After perm cols = ",cols)

U_in =[[[0,0] for i in list(range(N2))] for j in list(range(N2))]

for j in range(0,N2):
  temp_col = cols[j]
  b_in = np.random.randint(0, 2)
  #print("temp_col,j,b_in = ",temp_col,j,b_in)
  for i in range(0,N2):
    U_in[i][j] = U_temp[i][temp_col]
    if b_in == 1:
      U_in[i][j][0] = -U_in[i][j][0]

print("U_in = ",U_in)

#----------------------EXACT-V-OPT--------------------
#EXACT-V-OPT
  #Input channel rep
print("Input Unitary = ", U_in)
sde_in = GET_SDE(U_in,N2)
print("Input sde = ", sde_in)
m = sde_in
reach = 0

if m == 0:
    print("Not Clifford")

start_time = time.time()
while reach == 0:
    m_prime, D = EXACT_V_DECIDE(U_in,m,N2,sde_in)
    print("m_prime, D = ",m_prime,D)
    if m_prime == -1:
      m += 1
      #reach = 1
    else:
      print("m_prime, D = ", m_prime,D)
      print("Path length = ",len(D))
      print("Input Unitary = ", U_in)
      reach = 1

execTime = time.time()-start_time
print("Time = ",execTime)



