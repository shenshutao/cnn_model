import zipfile
import requests


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
    download_file_from_google_drive('1l_EmtfJ3QdH2S0QFhFsE22sMKn_vMAAn', 'train_data.zip')
    zip_ref = zipfile.ZipFile('train_data.zip', 'r')
    zip_ref.extractall('train_data')
    zip_ref.close()

    download_file_from_google_drive('1wKH68-hx8doGewV0n56l6G4j88tgowL7', 'pictures.zip')
    zip_ref = zipfile.ZipFile('pictures.zip', 'r')
    zip_ref.extractall('pictures')
    zip_ref.close()
