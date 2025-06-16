import os
import requests
from tqdm import tqdm
import shutil

def download(type):
  if (type == 'Data'):
    url = 'https://nlp.stanford.edu/projects/jesc/data/split.tar.gz'
    file_name = './Data/data.tar.gz'

    print('Starting download of data files')
    with open(file_name, 'wb') as file:
      response = requests.get(url)
      total_size = int(response.headers.get('content-length', 0))

      with tqdm(total=total_size, unit='kB', desc='Downloading') as progress_bar:
        for data in response.iter_content(chunk_size=1024):
          progress_bar.update(len(data))
          file.write(data)

    shutil.unpack_archive(file_name, f'./Data')
    os.remove(file_name)

  else:
    url = 'https://fonts.gstatic.com/s/notosansjp/v54/-F6jfjtqLzI2JPCgQBnw7HFyzSD-AsregP8VFBEj75vY0rw-oME.ttf'
    file_name = './Fonts/NotoSansJP.ttf'

    print('Starting download of font files')
    with open(file_name, 'wb') as file:
      response = requests.get(url)
      total_size = int(response.headers.get('content-length', 0))

      with tqdm(total=total_size, unit='kB', desc='Downloading') as progress_bar:
        for data in response.iter_content(chunk_size=1024):
          progress_bar.update(len(data))
          file.write(data)

def preprocess():
  if not os.path.exists('./Fonts'):
    os.makedirs('Fonts')
    
  if (len(os.listdir('./Fonts')) == 0):
    download('Fonts')
  else: print('Skipping download of font files')

  if not os.path.exists('./Data'):
    os.makedirs('Data')

  if (len(os.listdir('./Data')) == 0):
    download('Data')
  else: print('Skipping download of data files')

preprocess()