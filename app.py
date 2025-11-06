from vetiver import VetiverModel
from dotenv import load_dotenv, find_dotenv
import vetiver
import pins

load_dotenv(find_dotenv())

b = pins.board_gcs("info-4940-models/ji92/", allow_pickle_read=True)
PIN_NAME = "nhanes-prediabetes-py"
PIN_VERSION = "20251105T205309Z-32e04"

v = vetiver.VetiverModel.from_pin(b, PIN_NAME, version=PIN_VERSION)
api = vetiver.VetiverAPI(v, check_prototype=False)
app = api.app
