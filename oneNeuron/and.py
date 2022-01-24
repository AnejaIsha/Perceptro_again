from utils.model import Perceptron
from utils.all_utils import prepare_data,save_model,save_plot
import pandas as pd
import logging
import os

log_dir = "logs"
os.makedirs(log_dir,exist_ok = True)
logging_str = "[%(asctime)s:%(levelname)s:%(module)s] %(message)s"
logging.basicConfig(filename = os.path.join(log_dir,'logfile'),level = logging.INFO,format = logging_str,filemode ='a')


def main(data,modelName,plotName,eta,epochs):
    """[summary]

    Args:
        data ([type]): [description]
        modelName ([type]): [description]
        plotName ([type]): [description]
        eta ([type]): [description]
        epochs ([type]): [description]
    """

    df = pd.DataFrame(data)
    print(df)
    X, y = prepare_data(df)
    model = Perceptron(eta=eta, epochs=epochs)
    model.fit(X, y)
    _ = model.total_loss()
    save_model(model, filename=modelName)
    save_plot(df, plotName, model)


OR = {
"x1": [0,0,1,1],
"x2": [0,1,0,1],
"y": [0,1,1,1],
}
ETA = 0.3 # 0 and 1
EPOCHS = 10

try :
    main(data=OR, modelName="or.model", plotName="or.png", eta=ETA, epochs=EPOCHS)
except Exception as e:
    logging.info(e)
    print(e)

