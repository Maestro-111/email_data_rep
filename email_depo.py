import requests
from adal import AuthenticationContext

# Define your application's configuration
client_id = 'fc95af76-5c0f-4747-b2d6-78a3e5efa7f0'
client_secret = 'uWo8Q~MR4r5WExRmFe3J7aE5xtN1~5pRz1LK4dqs' 
authority_url = 'https://login.microsoftonline.com/d33b78bd-8756-4656-91ce-ffbb17c5cc22'
resource_url = 'https://outlook.office365.com'  
shared_mailbox_id = 'apcentralinvoices@gflenv.com'  # Replace with the shared mailbox's email address

# Authenticate and obtain an access token
context = AuthenticationContext(authority_url)
token = context.acquire_token_with_client_credentials(resource_url, client_id, client_secret)

access_token = token['accessToken']


# Set up the API endpoint for the shared mailbox
endpoint = f'https://graph.microsoft.com/v1.0/users/{shared_mailbox_id}/messages'

# Define headers with the access token
headers = {
    'Authorization': f'Bearer {access_token}',
}

# Make a GET request to retrieve messages from the shared mailbox
response = requests.get(endpoint, headers=headers)

print(response)

if response.status_code == 200:
    data = response.json()
    # Process the data (e.g., print the messages)
    for message in data.get('value', []):
        print(f"Subject: {message['subject']}")
        print(f"Received: {message['receivedDateTime']}")
    # You can access and process the message content as needed.
else:
    print(f"Failed to retrieve messages. Status code: {response.status_code}")


