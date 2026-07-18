import os
os.environ['KAGGLE_API_TOKEN'] = 'KGAT_90a00136263b9525e45b3f75d8273feb'

os.system('kaggle datasets download -d splcher/animefacedataset')
os.system('unzip animefacedataset.zip -d data/')