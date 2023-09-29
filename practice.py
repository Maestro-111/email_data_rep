def left_bound(ar,x):
    left,right = -1,len(ar)
    pos = -1
    while left < right - 1:
        mid = (left+right)//2
        if ar[mid] >= x:
            pos = mid
            right = mid
        else:
            left = mid
        
    return pos

    

def f_l(ar,x):
    first_pos = left_bound(ar,x)
    #last_pos = 1
    last_pos = left_bound(ar,x+1)-1
    print(first_pos,last_pos)
    if first_pos<=last_pos:
        return first_pos,last_pos
    return -1,-1
    print(first_pos,last_pos)



#ar = [5,7,7,8,8,10]
#x = 4
#print(f_l(ar,x))
import math

def ev(a,b):
    if a == 0 or b ==0:
        return max(a,b)
    answ = -math.inf
    for digit in range(1, min(a,b)+1):
        if a%digit == 0 and b%digit == 0 and digit >= answ:
            answ = digit
    return answ if answ != -math.inf else -1


#print(ev(32,1))
        
def min_conj(a,mod):
    interval = range(0,mod)

    for digit in interval:
        if (a-digit)%mod == 0:
            return digit



#print(min_conj(19,8))



def divide_two_strings(a,b):
    i = len(b)
    j = 0
    
    while i <= len(a):
        
        if a[j:i] == b:
            j = i
            i += len(b)
        else:
            break
            
    return a[j:]


s1 = 'AB'*1000000
s2 = 'ABC'


def gcdOfStrings(str1: str, str2: str) -> str:
        
    if len(str1) == len(str2):
        if str1 == str2:
            return str1
        return ''
        
    cand_min = min(str1, str2, key = len)
    cand_max = max(str1, str2, key = len)

        
    while len(cand_min) != 0 and len(cand_max) !=0:
            
        rem = divide_two_strings(cand_max,cand_min)

            
            
        if cand_max == rem:
            return ''
            
        c1 = cand_min
        c2 = rem
            
            
        if len(c1) == len(c2):
            cand_max = c1
            cand_min = c2
        else:
            cand_min = min(c1, c2, key = len)
            cand_max = max(c1, c2, key = len)

        #print(rem)
        #print(cand_max)
        #print(cand_min)
            
    return cand_min+cand_max
    

#print(gcdOfStrings(s1, s2))



def length_int(num):
    res = 0
    
    while num % 10:
        res += 1
        num = num // 10
    return res

def compress(chars: list[str]) -> int:
        
    res = 0
    last = None
    i = 0

    if len(chars) == 1:
        return 1
        
    while i < len(chars)-1:
        if chars[i] == chars[i+1]:
            c = 1
            
            if i+2 >= len(chars):
                res += 2
                break
            
            for j in range(i+2, len(chars)):
                if chars[j] != chars[i]:
                    break
                c += 1
                    
            i = j
            res += (1+length_int(c))
        else:
            res += 1
            i += 1
            
    if chars[-1] != chars[-2]:res+=1
    return res



    
#print(compress(["a","b",'c']))

  #0 1 3 6 10 15
 #[1,2,3,4,5]

def queries(ar,q):
    
    pref_sum = [0]+[None]*(len(ar))

  
    for i in range(len(ar)):
        pref_sum[i+1] = pref_sum[i]+ar[i]


    for a,b in q:
        right = pref_sum[b+1]
        left = pref_sum[a]
        print(right-left)



#queries([1,2,3,4,5],[[1,4]])


def merge_sorted_arrays(ar1, ar2):
    res = []
    
    i,j = 0,0
    
    
    while i < len(ar1) and j < len(ar2):
        if ar1[i] < ar2[j]:
            res.append(ar1[i])
            i += 1
        else:
            res.append(ar2[j])
            j += 1
    
    res = res + ar1[i:] + ar2[j:]


    return res


#print(merge_sorted_arrays([1, 5, 9, 10, 11, 13],[12,13,15]))


def check(s):
    i,j = 0,len(s)-1
    
    while i < j:
        if s[i] == s[j]:
            i += 1
            j -= 1
        else:
            return False
        
    return True



#print(check("aaacecaaacecaaa"))




def cycle(jump,D,start):

    res = 1 
    initial = start



    while start % D != 0:
        print(start)
        start = (start+jump)%D
        if start == initial:
            break
        res += 1



    return res

#print(cycle(8,36,12))



def min_p(grid):

    dp = [[0]*len(grid[0]) for i in range(len(grid))]

    dp[-1][-1] = grid[-1][-1]

    for i in range(len(grid[0])-2, -1, -1):
        dp[-1][i] = grid[-1][i]+dp[-1][i+1]

    for i in range(len(grid)-2, -1, -1):
        dp[i][-1] = grid[i][-1] + dp[i+1][-1]


    for i in range(len(grid)-2, -1, -1):
        for j in range(len(grid[0])-2, -1,-1):
            dp[i][j] = min(dp[i+1][j], dp[i][j+1])+grid[i][j]


    return dp[0][0]
        
    
#print(min_p( [[1,2,3],[4,5,6]]))



def vac(n, cost):

    dp = cost[0]

    for day in range(1,n):
        new_dp = [0]*3
        current_cost = cost[day]
        for i in range(len(dp)):
            for j in range(len(dp)):
                if i != j:
                    new_dp[j] = max(new_dp[j], dp[i]+current_cost[j])
        dp = new_dp


    return max(dp)


#print(vac(3, [[10,40,70],[20,50,80],[30,60,90]]))
import math

def ar_c(n):

    current_row_sum = lambda x : x*(x+1) // 2

    res = 0

    l = 1
    r = n

    while l <= r:

        mid = (l + r) // 2
        cur_sum = current_row_sum(mid)

        if cur_sum <= n:
            res = mid
            l = mid+1
        else:
            r = mid - 1

    return res

    

#print(ar_c(1100))





def accountsMerge(accounts):
    email_to_name = {}  # Mapping from email to name
    graph = {}  # Graph representation of accounts
        
    # Build the graph and email-to-name mapping
    for account in accounts:
        name = account[0]
        emails = account[1:]
        for email in emails:
            email_to_name[email] = name
            if email not in graph:
                graph[email] = set()
        for i in range(1, len(emails)):
            graph[emails[i-1]].add(emails[i])
            graph[emails[i]].add(emails[i-1])


    print(email_to_name)
    print(graph)
        
    visited = set()  # To keep track of visited emails
    result = []  # Final result list
        
    # DFS function to traverse the connected emails
    def dfs(email, current_account):
        current_account.append(email)
        visited.add(email)
        for neighbor in graph[email]:
            if neighbor not in visited:
                dfs(neighbor, current_account)
        
        # Iterate through each email and perform DFS if not visited
    for email in email_to_name:
        if email not in visited:
            current_account = []  # Store emails for the current account
            dfs(email, current_account)
            current_account.sort()  # Sort emails for the account
            result.append([email_to_name[email]] + current_account)
        
    return result

"""
x = [["David","David0@m.co","David4@m.co","David3@m.co"],["David","David5@m.co","David5@m.co","David0@m.co"],["David","David1@m.co","David4@m.co","David0@m.co"],
     ["David","David0@m.co","David1@m.co","David3@m.co"],["David","David4@m.co","David1@m.co","David3@m.co"]]
accountsMerge(x)
"""



import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
import random


def plot_points(jarvis):    
    def wrapper(trees):
        x_coords1, y_coords1 = zip(*trees)
        plt.subplot(1, 3, 1)
        plt.scatter(x_coords1, y_coords1, label='Data Points')
        obolochka = jarvis(trees)

        h = np.array(jarvis(trees))

        plt.subplot(1, 3, 2)
        plt.scatter(h[:, 0], h[:, 1], label='Data Points')
        hull = ConvexHull(h)
        
        for simplex in hull.simplices:
            plt.plot(h[simplex, 0], h[simplex, 1], 'k-')


        plt.subplot(1, 3, 3)
        x_coords2, y_coords2 = zip(*obolochka)
        plt.scatter(x_coords2, y_coords2, label='Data Points')
        
        
        plt.show()
        return obolochka    
    return wrapper

def find_angle(l1,l2):
    tang = l1/l2
    return math.degrees(math.atan(tang))


def length(vec):
    res = 0
    
    for coord in vec:
        res += coord**2
    
    return res**0.5



def dot(v1,v2):
    i  = 0
    res = 0
    
    while i < len(v1):
        res += (v1[i]*v2[i])
        i += 1
        
    return res


class get_dis:

    def __init__(self,initial_point):
        self.initial_point = initial_point
    
    def dis(self, point):
        i = 0
        res = 0
        while i < len(point):
            res += (point[i]-self.initial_point[i])**2
            i += 1

        return res**0.5


@plot_points
def outerTrees(trees):
        
    trees = sorted(trees)
    first_node = trees[0]
        
    x1,y1= first_node
        
    cand = (None,float('inf'))
        
    for i in range(1,len(trees)):
        x2,y2 = trees[i]
            
        if x2==x1:
            continue
            
        l1 = y2-y1
        l2 = x2-x1
        degree = find_angle(l1,l2)
            
        if cand[1] > degree:
            cand = ([x2,y2],degree)
                
        
    if cand[0] is None:
        return trees
        
    res = [first_node,cand[0]]
    print(res)
    first = res[0]
    last = res[-1] 

    while True:
            
        cand = (None,float('inf'))
        add = []
            
        for i in range(len(trees)):

            if trees[i] != first and trees[i] in res:
                continue
            
            x1,y1 = res[-2]
            x2,y2 = res[-1]
            x3,y3 = trees[i]

            v1 = (x1-x2,y1-y2)
            v2 = (x3-x2,y3-y2)

            print(trees[i])
            
            
            cos = round((dot(v1,v2))/(length(v1)*length(v2)),3)
            
            print(cos)
            
            if cos < cand[1]:
                cand = ([x3,y3],cos)
                add = []
                continue

            if cos == cand[1]:
                add.append([x3,y3])
                
        f_c = [cand[0]]
        current_res = f_c + add


        i_p = get_dis(last)
        

        current_res = sorted(current_res, key = i_p.dis)

        last = current_res[-1]
        res += current_res

        print(res)

        if last == first:
            res = list(map(lambda y : list(y), list(set(list(map(lambda x : tuple(x), res))))))
            break
        
    return res


def get_next_p(index, h):
    for i in range(index+1, len(h)):
        if not h[i]:
            return i+2




def countPrimes(n: int):
        
    if n <= 1:
        return 0
    if n == 2:
        return 1
    if n == 3:
        return 2
        
    h = [False for i in range(2,n)]
    start = 2
    prime = 2

    print(h)
        
    while prime**2 < n:
        for num in range(start,n,prime):
                
            h[num-2] = True

            if num == 2:
                h[0] = False

        
        prime = get_next_p(prime-2, h)
        start = prime**2


    print(h)
    
    return len(h)-sum(h)

print(countPrimes(7))
        

    






    
