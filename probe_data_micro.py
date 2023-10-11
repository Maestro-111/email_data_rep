import requests
import msal
import pandas as pd

#AP Central Invoices apcentralinvoices@gflenv.com
# setting up the connection to graph api

tenant_id = 'd33b78bd-8756-4656-91ce-ffbb17c5cc22'
client_id = 'fc95af76-5c0f-4747-b2d6-78a3e5efa7f0'
client_secret = 'uWo8Q~MR4r5WExRmFe3J7aE5xtN1~5pRz1LK4dqs' 
token_url = f'https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token'
resource_url = 'https://graph.microsoft.com'

app = msal.ConfidentialClientApplication(
    client_id=client_id,
    client_credential=client_secret,
    authority=f'https://login.microsoftonline.com/{tenant_id}'
)

scopes = ['https://graph.microsoft.com/.default']


result = app.acquire_token_silent(scopes=scopes, account=None)

if not result:
    result = app.acquire_token_for_client(scopes=scopes)


try:
    access_token = result['access_token']
except KeyError:
    raise Exception("no token")


headers = {
    'Authorization': 'Bearer ' + access_token,
    'Content-Type': 'application/json'
}


#######

class get_choice_int:
    def __init__(self,choices):
        self.choices = choices

    def make_choice(self):
        f = True
        try :
            a = int(input("Your choice: "))        
        except ValueError as v:
            f = False
        
        while not f or (a not in self.choices):
            print("Please, enter the correct input!")
            try :
                a = int(input("Your choice: "))
                f = True
            except ValueError as v:
                f = False
        return a


class get_choice_str:
    def __init__(self,choices):
        self.choices = choices

    def make_choice(self):
        a = input("Your choice: ").lower()
        
        while (a not in self.choices):
            print("Please, enter the correct input!")
            a = input("Your choice: ").lower()

            
        return a

    


def menu():
    print("%56s" % "------------------------------")
    print("%43s" % "Menu")
    print("%57s" % "------------------------------\n")
    print("Options: ")
    print("\nEnter 1, to search for the department")
    print("Enter 2, to search for the title")
    print("Enter 3, to search for the email (specific)")
    print("Enter 4, to exit")

    ch = get_choice_int([1,2,3,4])

    return ch.make_choice()

def main(): # get it run

    str_h = get_choice_str(['yes', 'no'])

    while True:
        
        choice = menu()
        
        if choice == 1:
            info = analyze_dep()
            write_to_excel(info)
            
        if choice == 2:
            info = analyze_title()
            write_to_excel(info)

        if choice == 3:
            info = analyze_email()
            write_to_excel(info)
            
        if choice == 4:
            break

        print("Would you like to continue? Say Yes or No\n")
        
        follow = str_h.make_choice()

        if follow == 'no':
            break
        
    print("Thank you")

    return




def get_info(query): # parsing output
    response = requests.get(query, headers=headers)

    users = response.json().get('value', [])

    name = []
    email = []
    officeLocation = []
    jobTitle = []
    dep = []
    
    for user in users:
        name.append(user['displayName'])
        email.append(user['mail'])
        officeLocation.append(user['officeLocation'])
        jobTitle.append(user['jobTitle'])
        dep.append(user['department'])

    users_matrix = pd.DataFrame({"Name":name, "Email":email, "Location":officeLocation, "Title":jobTitle, "Department":dep})

    users_matrix = users_matrix.sort_values(by='Name').reset_index(drop = True)
    
    return users_matrix




def get_info_2(query): # for emails and chart
    response = requests.get(query, headers=headers)

    if response.status_code == 200:
        data = response.json()

        # Get manager's information
        
        manager = {
            "displayName": data.get('displayName'),
            "mail": data.get('mail'),
            "officeLocation": data.get('officeLocation'),
            "jobTitle": data.get('jobTitle'),
            "department": data.get('department'),
            "userPrincipalName": data.get('userPrincipalName'),
            "region": data.get('region'),
            "businessUnitNumber": data.get('extension_b4b1e8de5793413b8f637c0dbec1f9ab_businessUnitNumber')
        }

        # Get direct reports
        direct_reports = data.get('directReports', [])

        #print(direct_reports)


        # Create a list to store information about direct reports
        direct_reports_info = []

        for user in direct_reports:
            direct_reports_info.append({
                "displayName": user.get('displayName'),
                "mail": user.get('mail'),
                "Location": user.get('officeLocation'),
                "jobTitle": user.get('jobTitle'),
                "department": user.get('department'),
                "Manager":data.get('displayName'),
                "Managers Email":data.get('mail')
            })

        manager_df = pd.DataFrame([manager])
        direct_reports_df = pd.DataFrame(direct_reports_info)

        return direct_reports_df
    
    else:
        print("Error:", response.status_code)
        return None


    
# queries

def analyze_dep(): # make anpother func to get data
    department_id = input("Enter your department: ")

    query = f"https://graph.microsoft.com/v1.0/users?$count=true&$filter=Department eq '{department_id}'&$select=displayName,mail,officeLocation,jobTitle,department"

    return get_info(query)


def analyze_title():
    title = input("Enter the title: ")

    query = f"https://graph.microsoft.com/v1.0/users?$count=true&$filter=jobTitle eq '{title}'&$select=displayName,mail,officeLocation,jobTitle,department"

    return get_info(query)


def analyze_email():
    email = input("Enter your email: ")

    query = f"https://graph.microsoft.com/v1.0/users/{email}?$expand=directReports($select=displayName,mail,officeLocation,jobTitle,department)"

    return get_info_2(query)




# pushing info to excel
    
def write_to_excel(info, path = 'C:/Users/ehuliiev/OneDrive - GFL Environmental Inc\Desktop/fill excel project/users dep.xlsx'):
    
    print("\nEnter 1, if you want to change the file path to excel sheet")
    print("Enter 2, if the end file is dep.xlsx (default file)")
    
    ch = get_choice_int([1,2])

    choice = ch.make_choice()

    if choice == 1:
        path = input("\nEnter your new path to the excel sheet")
    
    info.to_excel(path)

    print("\nLoaded the file to excel")

    return
    



main()


"""
upload_url = "https://graph.microsoft.com/v1.0/me/drive/root:/FolderName/AD Userlist Query.xlsx:/content"

# Read the Excel file from local disk
with open("path/to/your/FileName.xlsx", "rb") as file:
    file_contents = file.read()

# Upload the file to OneDrive
headers = {
    "Authorization": f"Bearer {access_token}",
    "Content-Type": "application/octet-stream"
}
response = requests.put(upload_url, data=file_contents, headers=headers)

if response.status_code == 201:
    print("File uploaded successfully.")
else:
    print("Error uploading file:", response.status_code, response.text)


"""




