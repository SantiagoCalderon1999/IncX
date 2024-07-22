import requests

def test_azure_deployment_check():
    # Given
    #url = 'https://incrementalexplainer.azurewebsites.net/'
    url = 'https://google.com' # TODO: Update the URL to the deployed web app
    
    # When
    response = requests.get(url)
    
    # Then
    assert response.status_code == 200, f"Web app is not deployed"
