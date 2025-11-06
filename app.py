from vetiver import VetiverModel
from dotenv import load_dotenv, find_dotenv
import vetiver
import pins

load_dotenv(find_dotenv())

b = pins.board_gcs('info-4940-models/ji92/', allow_pickle_read=True)
v = VetiverModel.from_pin(b, 'nhanes-prediabetes-py', version = '20251105T202433Z-7c0ea')

vetiver_api = vetiver.VetiverAPI(v)
api = vetiver_api.app
