from vetiver import VetiverModel
from dotenv import load_dotenv, find_dotenv
import vetiver
import pins
import sys 

def to_str_func(X):
    return X.astype(str)
sys.modules['__main__'].to_str_func = to_str_func

load_dotenv(find_dotenv())

b = pins.board_gcs('info-4940-models/ji92/', allow_pickle_read=True)
v = VetiverModel.from_pin(b, 'prediabetes-model-3', version = '20251106T183718Z-51f55')

vetiver_api = vetiver.VetiverAPI(v)
api = vetiver_api.app

