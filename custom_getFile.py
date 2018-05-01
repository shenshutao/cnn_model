import zipfile
import requests
import os

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


if __name__ == "__main__":
    #os.chdir('/Users/shutao/Desktop/abc')

    download_file_from_google_drive('12lMoH5HeZ61BjtUwSAMEdG3yxgnaAsGh', 'train_data.zip')
    zip_ref = zipfile.ZipFile('train_data.zip', 'r')
    zip_ref.extractall('train_data')
    zip_ref.close()

    download_file_from_google_drive('1N03DrCxcqDRjA5-YTEq098Gjf5C2imZ8', 'pictures.zip')
    zip_ref = zipfile.ZipFile('pictures.zip', 'r')
    zip_ref.extractall('pictures')
    zip_ref.close()
