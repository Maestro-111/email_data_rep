import pypyodbc
import pyodbc

DRIVER_NAME = 'SQL SERVER'
SERVER = 'GFLSQ15'
DATABASE = 'ETS'
USERNAME = 'GFL\ehuliiev'
PASSWORD = 'RETret6757%%%%'


c_s = f"""
    DRIVER={{DRIVER_NAME}};
    SERVER={SERVER};
    DATABASE={DATABASE};
    UID={USERNAME};
    PWD={PASSWORD};
    
"""

conn = pyodbc.connect(f"DRIVER={{SQL Server}};SERVER={SERVER};DATABASE={DATABASE};UID={USERNAME};PWD={PASSWORD}")

#f"DRIVER={SQL Server};SERVER={SERVER};DATABASE={DATABASE};UID={USERNAME};PWD={PASSWORD}"


class Solution:
    def findMaxAverage(self, nums, k: int) -> float:
        if len(nums) < k:
            return sum(nums[0:k])/len(nums[0:k])
        
        i,j = 0,k-1
        
        
        answ = sum(nums[0:k])
        current = answ
        
        
        while j < len(nums)-1:
            j += 1
            current = current - nums[i]+nums[j]
            answ = max(answ,current)
            i += 1
            
        return answ/k


#c = Solution()
#print(c.findMaxAverage([1,12,-5,-6,50,3,45,78,-100,32],4))








uid = 'GFL\ehuliiev'
pwd = 'RETret6757%%%%'
driver = "{SQL Server}"
server = "GFLSQ15"
database = "ETS"


conn = pyodbc.connect('DRIVER=' + driver + ';SERVER=' + server + '\SQLEXPRESS' + 
                      ';DATABASE=' + database + ';UID=' + uid + ';PWD=' + pwd + ';trusted_connection=yes')
